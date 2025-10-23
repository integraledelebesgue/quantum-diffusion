from collections.abc import Callable
from typing import Literal

import einops
import torch


class Diffusion(torch.nn.Module):
    prediction_goal: Literal["data", "noise"]
    height: int
    width: int
    directed: bool
    on_states: bool

    noise_function: Callable[..., torch.Tensor]

    net: torch.nn.Module
    loss: torch.nn.Module

    def __init__(
        self,
        net: torch.nn.Module,
        noise_function: Callable[..., torch.Tensor],
        prediction_goal: Literal["data", "noise"],
        shape: tuple[int, int],
        loss: torch.nn.Module = torch.nn.MSELoss(reduction="none"),
        directed: bool = False,
        on_states: bool = False,
    ) -> None:
        super().__init__()

        self.prediction_goal = prediction_goal
        self.width, self.height = shape
        self.directed = directed
        self.on_states = on_states

        self.noise_function = noise_function

        self.net = net
        self.loss = loss

        if self.on_states:
            self.loss = StateLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor, tau: int) -> torch.Tensor:
        whole_noisy = self.noise_function(x, tau=tau + 1, decay_mod=3.0)
        whole_noisy = einops.rearrange(
            whole_noisy, "(batch tau) pixels -> batch tau pixels", tau=tau + 1
        )

        batches_noisy = whole_noisy[:, 1:, :]
        batches_noisy = einops.rearrange(
            batches_noisy,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )

        batches_clean = whole_noisy[:, :-1, :]
        batches_clean = einops.rearrange(
            batches_clean,
            "batch tau (width height) -> (batch tau) 1 width height",
            width=self.width,
            height=self.height,
        )

        output = (
            self.net.forward(batches_noisy, y)
            if self.directed
            else self.net.forward(batches_noisy)
        )
        assert isinstance(output, torch.Tensor)

        match self.prediction_goal:
            case "data":
                batches_reconstructed = output
                batch_loss = self.loss(batches_reconstructed, batches_clean)
                batch_loss_mean = batch_loss.mean()

            case "noise":
                predicted_noise = output
                predicted_noise = (predicted_noise - 0.5) * 0.1
                real_noise = batches_noisy - batches_clean
                batch_loss = self.loss(predicted_noise, real_noise)
                batch_loss_mean = batch_loss.mean()

        return batch_loss_mean

    def sample(
        self,
        n_iters: int,
        first_x: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        show_progress: bool = False,
        only_last: bool = False,
        step: int = 1,
        noise_factor: float = 1.0,
    ) -> torch.Tensor:
        """ " Samples from the model for n_iters iterations."""
        if first_x is None:
            first_x = torch.rand((10, 1, self.width, self.height))
        if self.on_states:
            return self._sample_on_states(n_iters, first_x, only_last, labels=labels)

        if labels is None and self.directed:
            labels = torch.zeros((first_x.shape[0], 1))

        outp = [first_x]

        with torch.no_grad():
            x = first_x

            for i in range(n_iters):
                if self.directed:
                    predicted = self.net(x, labels)
                else:
                    predicted = self.net(x)

                if self.prediction_goal == "data":
                    x = predicted
                else:
                    predicted = (predicted - 0.5) * 0.1 * noise_factor
                    new_x = x - predicted
                    new_x = torch.clamp(new_x, 0, 1)
                    x = new_x

                if i % step == 0:
                    outp.append(x)

        if only_last:
            return outp[-1]
        else:
            outp = torch.stack(outp)
            outp = einops.rearrange(
                outp, "iters batch 1 height width -> (iters height) (batch width)"
            )

            return outp

    def _sample_on_states(
        self,
        n_iters: int,
        first_x: torch.Tensor,
        only_last: bool = True,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert only_last, "can't sample intermediate states, set `only_last=True`"
        assert self.prediction_goal == "data", "can't sample noise"
        assert self.on_states, "use sample() instead"
        return self.net.sample(first_x, num_repeats=n_iters, labels=labels)  # type: ignore

    def get_variance_sample(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the sample and the variance over the iterations.
        """
        sample = self.sample(**kwargs).abs()
        sample = einops.rearrange(
            sample,
            "(iters height) (batch width) -> iters batch height width",
            height=self.height,
            width=self.width,
        )
        vars = sample.var(dim=1)
        sample = einops.rearrange(
            sample, "iters batch height width -> (iters height) (batch width)"
        )
        vars = einops.rearrange(vars, "iters height width -> (iters height) (width)")
        return sample, vars

    def save_name(self):
        return f"{self.net.save_name()}{'_noise' if self.prediction_goal == 'noise' else ''}"  # type: ignore


class StateLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.is_complex(), "input must be complex"
        assert not target.is_complex(), "target must be real"
        return (input.real - target) ** 2 + input.imag**2
