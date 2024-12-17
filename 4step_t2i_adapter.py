import os
import time
import typing
from argparse import ArgumentParser, Namespace
from typing import Literal, Union

from diffusers import DiffusionPipeline, StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL, UNet2DConditionModel, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import load_image
from controlnet_aux.canny import CannyDetector
from controlnet_aux.zoe import ZoeDetector
from controlnet_aux.midas import MidasDetector
from huggingface_hub import hf_hub_download
from PIL import Image
import torch

PREPROCESSOR = Union[CannyDetector, ZoeDetector, MidasDetector, None]
PREPROCESSOR_NAME = Literal["canny", "depth_midas", "depth_zoe", "none"]


def prepare_pipe(
    t2i_adapter_name="TencentARC/t2i-adapter-canny-sdxl-1.0",
    **kwargs,
) -> DiffusionPipeline:

    # load adapter
    adapter = T2IAdapter.from_pretrained(t2i_adapter_name, torch_dtype=torch.float16, varient="fp16").to("cuda")

    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
    # Load model.
    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda"))

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_id, unet=unet, vae=vae, adapter=adapter, torch_dtype=torch.float16, variant="fp16", 
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe


def load_preprocessors(
    preprocessor_names: list[PREPROCESSOR_NAME] = ["canny"],
    **kwargs,
) -> list[PREPROCESSOR]:

    print(f"Loading ControlNet preprocessors: {preprocessor_names}")
    preprocessors = list()

    for name in preprocessor_names:
        if name == "canny":
            preprosessor = CannyDetector()

        elif name == "depth_midas":
            preprosessor = MidasDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        elif name == "depth_zoe":
            preprosessor = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        elif name == "none":
            preprosessor = None

        else:
            raise ValueError(f"Preprocessor name: '{name}' is not in {PREPROCESSOR_NAME}")

        preprocessors.append(preprosessor)

    return preprocessors


def preprocess_images(
    image: Image.Image,
    preprocessors: list[PREPROCESSOR],
    image_resolution=1024,
) -> list[Image.Image]:

    preprocessed_images: list[Image.Image] = list()

    for preprocessor in preprocessors:
        if isinstance(preprocessor, CannyDetector):
            # Detect the canny map in low resolution to avoid high-frequency details
            preprocessed_img = preprocessor(image, detect_resolution=384,
                                            image_resolution=image_resolution)#.resize((1024, 1024))

        elif isinstance(preprocessor, (MidasDetector, ZoeDetector)):
            preprocessed_img = preprocessor(image, image_resolution=image_resolution)

        elif preprocessor is None:
            preprocessed_img = image

        preprocessed_images.append(preprocessed_img)

    return preprocessed_images


def generate_images(
    pipe: DiffusionPipeline,
    preprocessors: list[PREPROCESSOR],
    image: str | Image.Image = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_canny.jpg",
    prompt = "Mystical fairy in real, magic, 4k picture, high quality",
    num_inference_steps=4,
    num_images_per_prompt=1,
    guidance_scale=0,
    adapter_conditioning_scale=0.8, 
    adapter_conditioning_factor=0.5,
    timesteps=[999, 749, 499, 249],
    **kwargs,
) -> list[Image.Image]:

    image = load_image(image)
    control_images = preprocess_images(image, preprocessors)

    begin = time.time()
    pipeline_output: StableDiffusionXLPipelineOutput = pipe(
        prompt=prompt,
        image=control_images,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
        adapter_conditioning_scale=adapter_conditioning_scale, 
        adapter_conditioning_factor=adapter_conditioning_factor,
        timesteps=timesteps,
    )
    end = time.time()
    print(f"Image generation time: {end - begin:.2f}sec")
    gen_images = pipeline_output.images
    
    return gen_images


def save_images(images: list[Image.Image], save_path="outputs/out_canny.png"):
    if len(images) == 0:
        raise ValueError("Input: `images` was empty.")
    if len(images) == 1:
        images[0].save(save_path)
    else:
        base, ext = os.path.splitext(save_path)
        for i, img in enumerate(images):
            save_path_numbered = f"{base}-{str(i)}{ext}"
            img.save(save_path_numbered)


def parse_kwargs(omit_none=True) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--preprocessor_names", type=str, nargs="+",
                        choices=typing.get_args(PREPROCESSOR_NAME)) # list[PREPROCESSOR_NAME]
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--num_inference_steps", "--steps", type=int)
    parser.add_argument("--num_images_per_prompt", "--batchsize", type=int)
    parser.add_argument("--guidance_scale", "--cfg_scale", type=float)

    args = parser.parse_args()
    kwargs = vars(args)

    if omit_none:
        kwargs = {key: val for key, val in kwargs.items() if val is not None}

    return kwargs


if __name__ == "__main__":
    print(f"{PREPROCESSOR_NAME = }")
    print(f"{vars(PREPROCESSOR_NAME) = }")
    print(f"{typing.get_args(PREPROCESSOR_NAME) = }")
    kwargs = parse_kwargs()
    print(f"kwargs:\n{kwargs}")

    pipe = prepare_pipe(**kwargs)
    preprocessors = load_preprocessors(**kwargs)
    gen_images = generate_images(pipe=pipe, preprocessors=preprocessors, **kwargs)

    save_images(gen_images)
