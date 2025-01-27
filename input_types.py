from dataclasses import dataclass, asdict

from PIL import Image

@dataclass
class ImageGenInput:
    image: Image.Image
    depth_image: Image.Image | None = None
    mask_image: Image.Image | None = None

    # TODO: For img2img=False (txt2img), the corresponding txt2img pipeline
    # has to be prepared, however, which is not yet.
    img2img: bool = True

    prompt: str = ""
    negative_prompt: str = ""

    # Denoising strength, between [0.0, 1.0].
    strength: float = 1.0

    # Fix it at 4 for DMD2, at least for now.
    num_inference_steps: int = 4

    # batchsize=8 caused OOM on A10G GPU on Modal only with 24 GB VRAM
    num_images_per_prompt: int = 1

    def as_dict(self):
        return asdict(self)
