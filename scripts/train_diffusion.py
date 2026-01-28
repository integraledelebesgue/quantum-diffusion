import einops
import grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import quantum_datasets
import quantum_diffusion
import quantum_image_encoding
import torch
import tqdm
from flax import nnx

jax.config.update('jax_enable_x64', True)

BATCH_SIZE = 64
NUM_DIFFUSION_STEPS = 10
NUM_EPOCHS = 100

torch.manual_seed(42)


def test(model: quantum_diffusion.neural_networks.pqc.PQCGuided, rngs: nnx.Rngs) -> None:
    noise = jax.dlpack.from_dlpack(torch.rand(15, 1, 8, 8) * 0.5 + 0.75)
    labels = jax.dlpack.from_dlpack(torch.randint(0, 2, (15,)))

    # labels = jax.random.bernoulli(rngs.test(), p=0.5, shape=(15,)).astype(jnp.float64)
    # noise = jax.random.uniform(rngs.test(), shape=(15, 1, 8, 8)) * 0.5 + 0.75

    results = [noise]

    for _ in range(2 * NUM_DIFFUSION_STEPS):
        images = model(noise, labels)
        results.append(images)
        noise = images

    results = jnp.stack(results)
    results = einops.rearrange(results, 'n b c h w -> c (n h) (b w)')

    plt.imshow(results.squeeze(0), cmap='gray')
    plt.axis('off')
    plt.savefig('data/test_plot.png')


@nnx.jit
def train_step(
    model: quantum_diffusion.neural_networks.pqc.PQCGuided,
    optimizer: nnx.Optimizer,
    images_noisy: jax.Array,
    images_clean: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    def loss_fn(model) -> jax.Array:
        images_denoised = model(images_noisy, labels)
        squared_error = (images_denoised - images_clean) ** 2
        return squared_error.mean()

    loss, gradient = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradient)
    return loss


def add_normal_noise_multiple(
    data: torch.Tensor,
    tau: int,
    decay_mod: float = 1.0,
) -> torch.Tensor:
    """
    Distorting the data by sampling from one normal distribution with a fixed mean.
    Adding the noise with different weights to the tensor.
    """

    batch, pixels = data.shape

    noise_weighting = torch.linspace(0, 1, tau) ** decay_mod
    noise_weighting = noise_weighting / noise_weighting.max()  # normalize
    noise_weighting = einops.repeat(noise_weighting, 'tau -> tau batch 1', batch=batch)

    noise = torch.normal(mean=0.5, std=0.2, size=(batch, pixels))
    noise_expanded = einops.repeat(noise, 'batch pixels -> tau batch pixels', tau=tau)

    data_expanded = einops.repeat(data, 'batch pixels -> tau batch pixels', tau=tau)

    noisy_data = data_expanded * (1 - noise_weighting) + noise_expanded * noise_weighting
    noisy_data = noisy_data.clamp(0, 1)
    noisy_data = einops.rearrange(noisy_data, 'tau batch pixels -> (batch tau) pixels')

    return noisy_data


def main() -> None:
    rngs = nnx.Rngs(params=0, noise=1, test=2)

    data_source = quantum_datasets.Digits8x8(num_classes=2)
    data_loader = (
        grain.MapDataset.source(data_source)
        .slice(slice(0, 80))
        .map(lambda data: (data[0] / 16, data[1].astype(jnp.float64)))
        .batch(BATCH_SIZE)
        .to_iter_dataset()
    )

    model = quantum_diffusion.neural_networks.pqc.PQCGuided(
        num_layers=16,
        input_shape=(1, 8, 8),
        encode=quantum_image_encoding.amplitude.encode,
        decode=quantum_image_encoding.amplitude.decode,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=1e-3, nesterov=False),
        wrt=nnx.Param,
    )

    progress_bar = tqdm.trange(NUM_EPOCHS, desc=f'[1/{NUM_EPOCHS}] Loss: -')

    for epoch in progress_bar:
        epoch_loss_total = jnp.array(0.0)

        for images, labels in data_loader:
            batch_size, num_channels, height, width = images.shape

            images_flat_tensor = torch.from_dlpack(images.__dlpack__()).flatten(-3, -1)
            whole_noisy = add_normal_noise_multiple(images_flat_tensor, NUM_DIFFUSION_STEPS + 1, 3.0)

            whole_noisy = einops.rearrange(whole_noisy, '(batch tau) pixels -> batch tau pixels', tau=NUM_DIFFUSION_STEPS + 1)
            batches_noisy = whole_noisy[:, 1:, :]
            batches_noisy = einops.rearrange(batches_noisy, 'batch tau (h w) -> (batch tau) 1 h w', h=8, w=8)
            images_noisy = jax.dlpack.from_dlpack(batches_noisy)

            batches_clean = whole_noisy[:, :-1, :]
            batches_clean = einops.rearrange(batches_clean, 'batch tau (h w) -> (batch tau) 1 h w', h=8, w=8)
            images_clean = jax.dlpack.from_dlpack(batches_clean)

            labels = einops.repeat(labels, 'b -> (b n)', n=NUM_DIFFUSION_STEPS)

            # noise_weights = jnp.linspace(0.0, 1.0, num=NUM_DIFFUSION_STEPS + 1) ** 3
            # noise_weights = noise_weights / noise_weights.max()
            # noise_weights = einops.repeat(noise_weights, 'n -> n b 1 1 1', b=batch_size)
            # image_weights = 1.0 - noise_weights

            # noise = jax.random.normal(rngs.noise(), images.shape)
            # noise = noise * 0.2 + 0.5
            # noise = noise / noise.max()
            # noise = einops.repeat(noise, 'b c h w -> n b c h w', n=NUM_DIFFUSION_STEPS + 1)

            # images = einops.repeat(images, 'b c h w -> n b c h w', n=NUM_DIFFUSION_STEPS + 1)
            # images_noisy = image_weights * images + noise_weights * noise
            # images_noisy = jnp.clip(images_noisy, 0.0, 1.0)

            # images_clean = images_noisy[:-1, ...]
            # images_clean = einops.rearrange(images_clean, 'n b c h w -> (n b) c h w')

            # images_noisy = images_noisy[1:, ...]
            # images_noisy = einops.rearrange(images_noisy, 'n b c h w -> (n b) c h w')

            # labels = einops.repeat(labels, 'b -> (n b)', n=NUM_DIFFUSION_STEPS)

            loss = train_step(model, optimizer, images_noisy, images_clean, labels)
            epoch_loss_total = epoch_loss_total + loss

        progress_bar.set_description(f'[{epoch + 1}/{NUM_EPOCHS}] Loss: {epoch_loss_total.item():.7f}')

    test(model, rngs)


if __name__ == '__main__':
    main()
