import einops
import grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import quantum_datasets
import quantum_diffusion
import quantum_image_encoding
import tqdm
from flax import nnx

BATCH_SIZE = 64
NUM_DIFFUSION_STEPS = 10
NUM_EPOCHS = 100
NUM_CLASSES = 2


def test(model: quantum_diffusion.neural_networks.pqc.PQCGuided, rngs: nnx.Rngs) -> None:
    labels = jax.random.randint(rngs.test(), shape=(15,), minval=0, maxval=NUM_CLASSES).astype(jnp.float32) / NUM_CLASSES
    noise = jax.random.uniform(rngs.test(), shape=(15, 1, 8, 8)) * 0.5 + 0.75
    noise = jnp.clip(noise, 0.0, 1.0)

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


@nnx.jit(static_argnames=['num_steps'])
def prepare_steps(
    images: jax.Array,
    labels: jax.Array,
    num_steps: int,
    decay: float,
    rngs: nnx.Rngs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size, *_ = images.shape

    noise_weights = jnp.linspace(0.0, 1.0, num=num_steps + 1) ** decay
    noise_weights = noise_weights / noise_weights.max()
    noise_weights = einops.repeat(noise_weights, 'n -> n b 1 1 1', b=batch_size)
    image_weights = 1.0 - noise_weights

    noise = jax.random.normal(rngs.noise(), images.shape)
    noise = noise * 0.2 + 0.5
    noise = einops.repeat(noise, 'b c h w -> n b c h w', n=num_steps + 1)

    images = einops.repeat(images, 'b c h w -> n b c h w', n=num_steps + 1)
    images_noisy = image_weights * images + noise_weights * noise
    images_noisy = jnp.clip(images_noisy, 0.0, 1.0)

    images_clean = images_noisy[:-1, ...]
    images_clean = einops.rearrange(images_clean, 'n b c h w -> (b n) c h w')

    images_noisy = images_noisy[1:, ...]
    images_noisy = einops.rearrange(images_noisy, 'n b c h w -> (b n) c h w')

    labels = einops.repeat(labels, 'b -> (b n)', n=num_steps)

    return images_clean, images_noisy, labels


def main() -> None:
    rngs = nnx.Rngs(params=0, noise=1, test=3)

    data_source = quantum_datasets.Digits8x8(num_classes=NUM_CLASSES)
    data_loader = (
        grain.MapDataset.source(data_source)
        .slice(slice(0, 80))
        .map(lambda data: (data[0] / 16, data[1].astype(jnp.float32) / NUM_CLASSES))
        .batch(BATCH_SIZE)
        .to_iter_dataset()
    )

    model = quantum_diffusion.neural_networks.pqc.PQCGuided(
        num_layers=16,
        input_shape=(1, 8, 8),
        encoding=quantum_image_encoding.FRQI((1, 8, 8)),
        device='lightning.qubit',
        qjit=True,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=1e-3, nesterov=True),
        wrt=nnx.Param,
    )

    progress_bar = tqdm.trange(NUM_EPOCHS, desc=f'[1/{NUM_EPOCHS}] Loss: -')

    for epoch in progress_bar:
        epoch_loss_total = jnp.array(0.0)

        for images, labels in data_loader:
            images_clean, images_noisy, labels = prepare_steps(images, labels, num_steps=NUM_DIFFUSION_STEPS, decay=3.0, rngs=rngs)
            loss = train_step(model, optimizer, images_noisy, images_clean, labels)
            epoch_loss_total = epoch_loss_total + loss

        progress_bar.set_description(f'[{epoch + 1}/{NUM_EPOCHS}] Loss: {epoch_loss_total.item():.7f}')

    test(model, rngs)


if __name__ == '__main__':
    main()
