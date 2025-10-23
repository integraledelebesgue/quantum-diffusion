import pathlib
import webbrowser

import click
import matplotlib.pyplot as plt
import torch
import tqdm
from loguru import logger

from . import data, models, nn, noise


def train(
    diff: models.Diffusion,
    ds: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
    epochs: int,
    tau: int,
    lr: float,
    save_path: pathlib.Path,
) -> None:
    logger.info("Training model")
    diff.train()

    pbar = tqdm.tqdm(total=epochs)
    opt = torch.optim.Adam(diff.parameters(), lr=lr)

    for _ in range(epochs):
        epoch_loss = 0.0

        x: torch.Tensor
        y: torch.Tensor
        for x, y in ds:
            opt.zero_grad()
            batch_loss = diff.forward(x, y, tau)
            epoch_loss += batch_loss.mean()
            opt.step()

        pbar.set_postfix({"loss": epoch_loss.item()})  # type: ignore
        pbar.update(1)

    pbar.close()
    sp = save_path / f"{diff.save_name()}.pt"

    if not sp.parent.exists():
        sp.parent.mkdir(parents=True)

    torch.save(diff.state_dict(), sp)


def test(diff: models.Diffusion, tau: int, save_path: pathlib.Path) -> None:
    logger.info("Testing model")
    diff.eval()
    first_x = torch.rand(15, 1, 8, 8) * 0.5 + 0.75
    outp = diff.sample(first_x=first_x, n_iters=tau * 2, show_progress=True)
    plt.imshow(outp.cpu(), cmap="gray")
    plt.axis("off")
    sp = save_path / f"{diff.save_name()}.png"
    plt.savefig(sp)
    webbrowser.open(sp.absolute().as_uri())


@click.command(help="Quantum Denoising Diffusion Model CLI")
@click.option(
    "--dataset",
    type=str,
    default="mnist_8x8",
    show_default=True,
    help="Dataset to use",
)
@click.option(
    "--dataset-location",
    type=pathlib.Path,
    help="Location to download the dataset to",
)
@click.option(
    "--n-classes",
    type=int,
    default=2,
    show_default=True,
    help="Number of label classes to use. Smaller models perform better on a smaller number of classes.",
)
@click.option(
    "--target",
    type=click.Choice(["noise", "data"], case_sensitive=False),
    default="noise",
    show_default=True,
    help="Generate 'noise' or 'data'.",
)
@click.option(
    "--save-path",
    type=pathlib.Path,
    default="results",
    show_default=True,
    help="Path to save results.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "--load-path",
    type=pathlib.Path,
    default=None,
    help="Path to load model from. If not provided, a new model will be trained and saved in --save-path.",
)
@click.option(
    "--model",
    type=str,
    default="QDenseUndirected",
    show_default=True,
    help=("Model name"),
)
@click.option(
    "--model-parameters",
    type=str,
    default="(8, 55)",
    help="Constructor parameters for the model",
)
@click.option(
    "--guidance/--no-guidance",
    default=False,
    show_default=True,
    help="Toggle guidance on or off.",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda", "mps"], case_sensitive=False),
    show_default=True,
    help="Device to use.",
)
@click.option(
    "--tau",
    type=int,
    default=10,
    show_default=True,
    help="Number of iterations (tau). Higher values work better for higher resolution images.",
)
@click.option(
    "--ds-size",
    type=int,
    default=100,
    show_default=True,
    help="Dataset size (80% of this will be used for training).",
)
@click.option(
    "--lr",
    type=float,
    default=1e-4,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--epochs",
    type=int,
    default=10000,
    show_default=True,
    help="Number of training epochs.",
)
def main(
    dataset: str,
    dataset_location: pathlib.Path | None,
    n_classes: int,
    target: str,
    save_path: pathlib.Path,
    seed: int | None,
    load_path: pathlib.Path | None,
    model: str,
    model_parameters: str,
    guidance: bool,
    device: str,
    tau: int,
    ds_size: int,
    lr: float,
    epochs: int,
) -> None:
    if seed is not None:
        torch.manual_seed(seed)

    if device == "cuda":
        logger.warning("CUDA performance is worse than CPU for most models.")
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, using CPU.")
            device = "cpu"

    x_train, y_train, height, width = data.get_by_name(
        dataset,
        dataset_location,
        n_classes,
        ds_size,
    )

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    train_cutoff = int(len(x_train) * 0.8)

    x_train, x_test = x_train[:train_cutoff], x_train[train_cutoff:]
    y_train, y_test = y_train[:train_cutoff], y_train[train_cutoff:]

    ds = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=10,
        shuffle=False,
    )

    net = nn.get_by_name(model, eval(model_parameters))

    diff = models.Diffusion(
        net=net,
        shape=(height, width),
        noise_function=noise.add_normal_noise_multiple,
        prediction_goal=target,
        directed=guidance,
        loss=torch.nn.MSELoss(),
    ).to(device)

    run_train = False

    if load_path is not None:
        # print("Loading model")
        try:
            if load_path.suffix == ".pt":
                diff.load_state_dict(torch.load(load_path))
            else:
                lp = load_path / f"{diff.save_name()}.pt"
                diff.load_state_dict(torch.load(lp))

        except FileNotFoundError:
            logger.error("Failed to load model")
            run_train = True

    if load_path is None or run_train:
        train(diff, ds, epochs, tau, lr, save_path)

    test(diff, tau, save_path)


if __name__ == "__main__":
    main()
