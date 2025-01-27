import time
from argparse import ArgumentParser, Namespace

import modal
from PIL import Image

from input_types import ImageGenInput


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--image", type=str,
                        default="inputs/color_armor.png",
                        help="Path to the input image file.")
    parser.add_argument("--mask_image", type=str,
                        default="inputs/mask_armor.png",
                        help="Path to the mask image file.")
    parser.add_argument("--prompt", type=str,
                        default="Medieval knight armor, plain background, 4K HDR photo, high quality, masterpiece, extremely detailed, sharp, clear")
    parser.add_argument("--batchsize", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Not directly importing `ImageGen()` to
    # avoid long library import time and dependencies.
    ImageGen = modal.Cls.lookup("DMD2", "ImageGen")
    modal_image_gen = ImageGen()

    color_image = Image.open(args.image)
    mask_image = Image.open(args.mask_image).convert("L")
    print(f"image_path = {args.image}, {color_image.size = }, {color_image.mode = }")
    print(f"mask_image_path = {args.mask_image}, {mask_image.size = }, {mask_image.mode = }")

    inputs = ImageGenInput(
        img2img=True,
        image=color_image,
        mask_image=mask_image,
        prompt=args.prompt,
        batchsize=args.batchsize,
    )
    print(f"{inputs = }")

    begin_image_gen = time.time()

    # Calling Modal function
    gen_images = modal_image_gen.run.remote(inputs=inputs)

    end_image_gen = time.time()
    print(f"Received {len(gen_images)} image(s) of size: {gen_images[0].size}.")
    print(f"Image generation time (Client side): {end_image_gen - begin_image_gen:.2f} sec.")

if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    print(f"Overall time: {end - begin:.2f} sec.")
