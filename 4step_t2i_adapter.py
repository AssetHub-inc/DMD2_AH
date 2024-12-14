from diffusers import DiffusionPipeline, StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image
from controlnet_aux.canny import CannyDetector
from huggingface_hub import hf_hub_download
from PIL import Image
import torch


def prepare_pipe(
    t2i_adapter_name="TencentARC/t2i-adapter-canny-sdxl-1.0",
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


def generate_image(
    pipe: DiffusionPipeline,
    image: str | Image.Image = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_canny.jpg",
    prompt = "Mystical fairy in real, magic, 4k picture, high quality",
    num_inference_steps=4,
    guidance_scale=0,
    adapter_conditioning_scale=0.8, 
    adapter_conditioning_factor=0.5,
    timesteps=[999, 749, 499, 249],
    save_path="outputs/out_canny.png",
):

    image = load_image(image)
    canny_detector = CannyDetector()

    # Detect the canny map in low resolution to avoid high-frequency details
    image = canny_detector(image, detect_resolution=384, image_resolution=1024)#.resize((1024, 1024))

    gen_images: Image.Image = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        adapter_conditioning_scale=adapter_conditioning_scale, 
        adapter_conditioning_factor=adapter_conditioning_factor,
        timesteps=timesteps,
    ).images[0]
    gen_images.save(save_path)


if __name__ == "__main__":
    pipe = prepare_pipe()
    generate_image(pipe=pipe)
