import os
import time
from argparse import ArgumentParser, Namespace

from diffusers import DiffusionPipeline, StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL, UNet2DConditionModel, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import load_image
from controlnet_aux.canny import CannyDetector
from huggingface_hub import hf_hub_download
from PIL import Image
import torch


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


def generate_images(
    pipe: DiffusionPipeline,
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
    canny_detector = CannyDetector()

    # Detect the canny map in low resolution to avoid high-frequency details
    image = canny_detector(image, detect_resolution=384, image_resolution=1024)#.resize((1024, 1024))

    begin = time.time()
    pipeline_output: StableDiffusionXLPipelineOutput = pipe(
        prompt=prompt,
        image=image,
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

    parser.add_argument("--prompt", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--num_inference_steps", "--steps", type=int)
    parser.add_argument("--num_images_per_prompt", "--batchsize", type=int)

    args = parser.parse_args()
    kwargs = vars(args)

    if omit_none:
        kwargs = {key: val for key, val in kwargs.items() if val is not None}

    return kwargs


if __name__ == "__main__":
    kwargs = parse_kwargs()
    print(f"kwargs:\n{kwargs}")

    pipe = prepare_pipe(**kwargs)
    gen_images = generate_images(pipe=pipe, **kwargs)

    save_images(gen_images)
