import PIL
import mlx.core as mx
from PIL import Image
from tqdm import tqdm

from flux_1.config.config_img2img import ConfigImg2Img
from flux_1.config.model_config import ModelConfig
from flux_1.config.runtime_config import RuntimeConfig
from flux_1.flux import Flux1
from flux_1.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1.models.transformer.transformer import Transformer
from flux_1.models.vae.vae import VAE
from flux_1.post_processing.image_util import ImageUtil
from flux_1.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1.tokenizer.t5_tokenizer import TokenizerT5
from flux_1.tokenizer.tokenizer_handler import TokenizerHandler
from flux_1.weights.weight_handler import WeightHandler


class Flux1Img2Img:

    def __init__(self, repo_id: str):
        self.model_config = ModelConfig.from_repo(repo_id)

        # Initialize the tokenizers
        tokenizers = TokenizerHandler.load_from_disk_or_huggingface(repo_id, self.model_config.max_sequence_length)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        weights = WeightHandler.load_from_disk_or_huggingface(repo_id)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)

    @staticmethod
    def from_repo(repo_id: str) -> "Flux1Img2Img":
        return Flux1Img2Img(repo_id)

    @staticmethod
    def from_alias(alias: str) -> "Flux1Img2Img":
        return Flux1Img2Img(ModelConfig.from_alias(alias).model_name)

    def generate_image(
            self,
            seed: int,
            prompt: str,
            image_path: str,
            config: ConfigImg2Img
    ) -> PIL.Image.Image:
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)

        # 0. Load the base image and let it determine the height and width for the generation
        base_image = ImageUtil.load_image(image_path)
        base_image = ImageUtil.to_array(base_image)
        config.config.height = base_image.shape[2]
        config.config.width = base_image.shape[3]

        # 1. Create the latents (based on a mix of random noise and the base image)
        image_latents = self.vae.encode(base_image)
        image_latents = self._pack_latents(image_latents, config.height, config.width)
        initial_noise = mx.random.normal(shape=image_latents.shape, key=mx.random.key(seed))
        sigma = config.sigmas[config.config.init_timestep]
        latents = sigma * initial_noise + (1.0 - sigma) * image_latents

        # 2. Embedd the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for t in tqdm(config.num_inference_steps, desc="Generating image", unit="step"):
            # 3.t Predict the noise
            noise = self.transformer.predict(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config,
            )

            # 4.t Take one denoise step
            dt = config.sigmas[t + 1] - config.sigmas[t]
            latents += noise * dt

            # To enable progress tracking
            mx.eval(latents)

        # 5. Decode the latent array
        latents = Flux1.unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)

    @staticmethod
    def _pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, 16, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        latents = mx.reshape(latents, (1, (width // 16) * (height // 16), 64))
        return latents
