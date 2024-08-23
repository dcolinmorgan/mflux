import logging

import mlx.core as mx

log = logging.getLogger(__name__)


class ConfigImg2Img:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_inference_steps: int = 4,
            guidance: float = 4.0,
            strength: float = 0.5,
    ):
        self._validate_input(strength)

        self._width = None
        self._height = None
        self.guidance = guidance
        self.strength = strength
        self.num_total_denoising_steps = int(num_inference_steps / (1 - strength))
        self.init_timestep = int(self.num_total_denoising_steps - num_inference_steps)
        self.num_inference_steps = list(range(self.num_total_denoising_steps))[self.init_timestep:]

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @staticmethod
    def _validate_input(strength: float) -> None:
        if strength <= 0.0 or strength >= 1.0:
            raise ValueError("Strength should be a float between 0 and 1.")
