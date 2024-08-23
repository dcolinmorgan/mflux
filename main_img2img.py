import argparse
import os
import sys
import time

from flux_1.config.config_img2img import ConfigImg2Img
from flux_1.flux_img2img import Flux1Img2Img

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.post_processing.image_util import ImageUtil


def main():
    parser = argparse.ArgumentParser(description='Generate an image based on a prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='The textual description of the image to generate.')
    parser.add_argument('--base-image', type=str, required=True, help='The path to the base image.')
    parser.add_argument('--output', type=str, default="image.png", help='The filename for the output image. Default is "image.png".')
    parser.add_argument('--model', type=str, default="schnell", help='The model to use ("schnell" or "dev"). Default is "schnell".')
    parser.add_argument('--seed', type=int, default=None, help='Entropy Seed (Default is time-based random-seed)')
    parser.add_argument('--steps', type=int, default=4, help='Inference Steps')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance Scale (Default is 3.5)')
    parser.add_argument('--strength', type=float, default=0.5, help='Regulates much the final image should be influenced by base image. Default is 0.5.')

    args = parser.parse_args()

    seed = int(time.time()) if args.seed is None else args.seed

    flux = Flux1Img2Img.from_alias(args.model)

    image = flux.generate_image(
        seed=seed,
        prompt=args.prompt,
        image_path=args.base_image,
        config=ConfigImg2Img(
            num_inference_steps=args.steps,
            guidance=args.guidance,
            strength=args.strength,
        )
    )

    ImageUtil.save_image(image, args.output)


if __name__ == '__main__':
    main()
