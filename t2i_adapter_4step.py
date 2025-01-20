import time
begin_import = time.time()
print("Importing libraries...")

import asyncio
import os
import typing
from argparse import ArgumentParser, Namespace
from typing import Literal, Union

from diffusers import DiffusionPipeline, StableDiffusionXLAdapterPipeline, T2IAdapter, MultiAdapter, AutoencoderKL, UNet2DConditionModel, LCMScheduler
from diffusers.models import ControlNetModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import load_image
from controlnet_aux.canny import CannyDetector
from controlnet_aux.zoe import ZoeDetector
from controlnet_aux.midas import MidasDetector
from huggingface_hub import hf_hub_download
from PIL import Image
import torch

from pipeline_stable_diffusion_xl_controlnet_adapter_inpaint import StableDiffusionXLControlNetAdapterInpaintPipeline

end_import = time.time()
print(f"Library import time: {end_import - begin_import:.2f} sec")


# Orders of T2I_ADAPTER_NAME and T2I_ADAPTER_FULLNAME should match
T2I_ADAPTER_NAME = Literal["canny", "depth_midas", "depth_zoe"]
T2I_ADAPTER_FULLNAME = Literal[
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
    "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
]
T2I_ADAPTER_NAME_TO_FULLNAME = {
    name: fullname for name, fullname in
    zip(typing.get_args(T2I_ADAPTER_NAME), typing.get_args(T2I_ADAPTER_FULLNAME))
}
print(f"{T2I_ADAPTER_NAME_TO_FULLNAME = }")

PREPROCESSOR = Union[CannyDetector, ZoeDetector, MidasDetector, None]
PREPROCESSOR_NAME = Literal["canny", "depth_midas", "depth_zoe", "none"]


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function '{func.__name__}()' is running...")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}()' took {end_time - start_time:.2f} sec.")
        return result
    return wrapper


def measure_async_execution_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function '{func.__name__}()' is running...")
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}()' took {end_time - start_time:.2f} sec.")
        return result
    return wrapper


def load_t2i_adapter(name: T2I_ADAPTER_NAME):
    fullname = T2I_ADAPTER_NAME_TO_FULLNAME[name]
    return T2IAdapter.from_pretrained(fullname, torch_dtype=torch.float16, varient="fp16").to("cuda")


@measure_execution_time
def load_adapters(t2i_adapter_names: list[T2I_ADAPTER_NAME] = ["canny"]
                  ) -> T2IAdapter | MultiAdapter:

    adapters = [load_t2i_adapter(name) for name in t2i_adapter_names]
    if len(adapters) == 1:
        return adapters[0]
    else:
        return MultiAdapter(adapters)


def load_loras_to_pipe(
    pipe: DiffusionPipeline | StableDiffusionXLAdapterPipeline | StableDiffusionXLControlNetAdapterInpaintPipeline,
    loras: list[str],
    lora_weights=1.0,
    lora_dir="/workspace/models/loras/",
) -> DiffusionPipeline | StableDiffusionXLAdapterPipeline | StableDiffusionXLControlNetAdapterInpaintPipeline:

    if isinstance(lora_weights, (int, float)):
        lora_weights = [float(lora_weights)] * len(loras)
    elif len(loras) != len(lora_weights):
        raise ValueError("The lengths of `lora_weights` should be the same as "
                         + f"that of `loras` but {len(lora_weights)} were given.")

    adapter_names = list()
    for lora in loras:
        adapter_name = os.path.splitext(lora)[0]
        adapter_names.append(adapter_name)
        pipe.load_lora_weights(lora_dir, weight_name=lora, adapter_name=adapter_name)

    pipe.set_adapters(adapter_names, adapter_weights=lora_weights)
    pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
    pipe.unload_lora_weights()

    return pipe


DMD2_TIMESTEPS = [999, 749, 499, 249]

def inject_timesteps_into_scheduler(scheduler_cls: SchedulerMixin, timesteps=DMD2_TIMESTEPS):
    class SchedulerWithFixedTimesteps(scheduler_cls):
        def __init__(self): 
            super().__init__()

        # Inject hard coded timesteps for DMD2
        # `num_inference_steps` will be ignored
        def set_timesteps(self, num_inference_steps: int | None = 4, **kwargs):
            print(f"{num_inference_steps = }, which will be ignored.")
            default_timesteps = kwargs.get("timesteps", None)
            print(f"Replacing the given {default_timesteps = } with {timesteps = }.")
            super().set_timesteps(timesteps=timesteps, **kwargs)

    return SchedulerWithFixedTimesteps


@measure_execution_time
def load_vae(vae_name="madebyollin/sdxl-vae-fp16-fix"):
    return AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float16)


@measure_async_execution_time
async def load_DMD2_unet(
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
) -> UNet2DConditionModel:

    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

    # Load SDXL UNet model weights first.
    unet = UNet2DConditionModel.from_config(
        config=UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    )
    # ).to("cuda", torch.float16)

    # Load DMD2 UNet weights.
    unet.load_state_dict(torch.load(
        hf_hub_download(repo_name, ckpt_name),
        map_location="cuda",
        weights_only=True,
    ))

    return unet


@measure_execution_time
def load_controlnet(model_name="diffusers/controlnet-depth-sdxl-1.0", disable=False):
    if disable:
        return None

    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # ).to("cuda")

    return controlnet


@measure_async_execution_time
async def load_DMD2_pipe(
    t2i_adapter_names: list[T2I_ADAPTER_NAME] = ["canny"],
    img2img=False,
    loras: list[str] | None = None,
    lora_weights=1.0,
    inject_timesteps=False,
    **kwargs,
) -> DiffusionPipeline:

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    begin_load = time.time()
    print("Loading T2I Adapters, VAE, DMD2 UNet, and ControlNet...")

    begin_tasks = time.time()
    print(f"Starting tasks (load T2I Adapters, VAE, and ControlNet)...")

    # Start tasks that will run concurrently.
    # TODO: Fix the orders and concurrency of async tasks.
    # Somehow, print debugs told that `load_DMD2_unet()`
    # still starts and finishes earlier than these async tasks .
    tasks = [
        asyncio.create_task(asyncio.to_thread(load_adapters, t2i_adapter_names)),
        asyncio.create_task(asyncio.to_thread(load_vae)),
        # Disable ControlNet if txt2img (not img2img) as not needed by the pipeline
        asyncio.create_task(asyncio.to_thread(load_controlnet, disable=not img2img)),
    ]

    # Start and await UNet loading first as it takes much longer.
    unet = await load_DMD2_unet(base_model_id=base_model_id)
    adapter, vae, controlnet_depth = await asyncio.gather(*tasks)

    end_tasks = time.time()
    print(f"Threaded task execution time (T2I Adapters, VAE, and ControlNet): {end_tasks - begin_tasks:.2f} sec")

    controlnet_depth = controlnet_depth.to("cuda", torch.float16)
    unet = unet.to("cuda", torch.float16)

    end_load = time.time()
    print(f"Loading time (T2I Adapters, VAE, DMD2 UNet, and ControlNet): {end_load - begin_load:.2f} sec")

    if img2img:
        # StableDiffusionXLControlNetAdapterInpaintPipeline
        # https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#controlnet--t2i-adapter--inpainting-pipeline
        # https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_stable_diffusion_xl_controlnet_adapter_inpaint.py
        pipe = StableDiffusionXLControlNetAdapterInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=base_model_id,
            unet=unet,  # DMD2
            vae=vae,
            adapter=adapter,
            controlnet=controlnet_depth,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    else: # txt2img
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            base_model_id, unet=unet, vae=vae, adapter=adapter, torch_dtype=torch.float16, variant="fp16", 
        ).to("cuda")

    print(f"Loaded diffusers pipeline: {type(pipe)}")

    if loras is not None:
        print(f"Loading LoRAs...: {loras}")
        load_loras_to_pipe(pipe=pipe, loras=loras, lora_weights=lora_weights)

    if inject_timesteps:
        scheduler_cls = inject_timesteps_into_scheduler(LCMScheduler)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    print(f"Device at the end of 'load_DMD2_pipe()' is: {pipe.device}")

    return pipe


@measure_async_execution_time
async def load_preprocessors(
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


@measure_execution_time
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


def create_all_white_mask(size=tuple[int]):
    r"""
    For the argument `mask_image` of an inpaint pipeline,
    such as StableDiffusionXLControlNetAdapterInpaintPipeline,
    White pixels in the mask will be repainted,
    while black pixels will be preserved.
    """
    assert len(size) == 2, size

    return Image.new(mode="I", size=size, color=255)  # Grayscale, white (255)


@measure_execution_time
def generate_images(
    pipe: DiffusionPipeline,
    preprocessors: list[PREPROCESSOR],
    image: str | Image.Image = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_canny.jpg",
    mask_image: str | Image.Image | None = None,
    prompt = "Mystical fairy in real, magic, 4k picture, high quality",
    negative_prompt="",
    num_inference_steps=4,
    num_images_per_prompt=1,
    guidance_scale=0,
    strength=1.0,  # Denoising strength
    adapter_conditioning_scale: float | list[float] = 0.8,
    img2img=False,
    adapter_conditioning_factor=0.5,  # Ignored if `is_inpaint` is True
    timesteps=[999, 749, 499, 249],   # Ignored if `is_inpaint` is True
    **kwargs,
) -> list[Image.Image]:

    image = load_image(image)

    if img2img:
        if mask_image is None:
            mask_image = create_all_white_mask(image.size)
        else:
            mask_image = load_image(mask_image)

    control_images = preprocess_images(image, preprocessors)

    if len(control_images) == 1: # Single T2I Adapter
        # It needs to be a single float value
        if isinstance(adapter_conditioning_scale, list):
            # list[float] (len == 1) -> float
            adapter_conditioning_scale = adapter_conditioning_scale[0]
    else: # Multiple T2I Adapters
        if isinstance(adapter_conditioning_scale, (float, int)):
            adapter_conditioning_scale = [adapter_conditioning_scale]
        # If original input is float or list[float] of length 1.
        # Skip otherwise (multiple scales are already provided, which will be used directly)
        if len(adapter_conditioning_scale) == 1:
            adapter_conditioning_scale = adapter_conditioning_scale * len(control_images)

    begin = time.time()
    print(f"Generating {num_images_per_prompt} images...")

    if img2img:
        # StableDiffusionXLControlNetAdapterInpaintPipeline
        pipeline_output: StableDiffusionXLPipelineOutput = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            adapter_image=control_images,
            control_image=control_images[-1],
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            strength=strength,  # Denoising strength
            adapter_conditioning_scale=adapter_conditioning_scale,
            controlnet_conditioning_scale=1.0,
            control_guidance_end=1e-8,  # Do NOT apply ControlNet at all, cannot be exactly 0.0
            width=1024,
            height=1024,
        )

    else:
        # txt2img with T2I Adapter, StableDiffusionXLAdapterPipeline
        pipeline_output: StableDiffusionXLPipelineOutput = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_images,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            timesteps=timesteps,
        )

    end = time.time()
    print(f"Image generation time: {end - begin:.2f} sec")
    gen_images = pipeline_output.images
    
    return gen_images


async def load_pipe_and_preprocessors(**kwargs):
    pipe, preprocessors = await asyncio.gather(
        load_DMD2_pipe(**kwargs),
        load_preprocessors(**kwargs),
    )
    pipe = pipe.to("cuda")
    print(f"Device is: {pipe.device}")

    return pipe, preprocessors


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
    parser.add_argument("--t2i_adapter_names", type=str, nargs="+",
                        choices=typing.get_args(T2I_ADAPTER_NAME)) # list[T2I_ADAPTER_NAME]
    parser.add_argument("--preprocessor_names", type=str, nargs="+",
                        choices=typing.get_args(PREPROCESSOR_NAME)) # list[PREPROCESSOR_NAME]
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--mask_image", type=str)  # Inpaint mask
    parser.add_argument("--img2img", action="store_true")  # Default: False
    parser.add_argument("--num_inference_steps", "--steps", type=int)
    parser.add_argument("--num_images_per_prompt", "--batchsize", type=int)
    parser.add_argument("--guidance_scale", "--cfg_scale", type=float)
    parser.add_argument("--strength", "--denoising_strength", type=float, default=1.0)
    parser.add_argument("--adapter_conditioning_scale", type=float, nargs="*") # list[float], Optional
    parser.add_argument("--loras", type=str, nargs="*")  # list[str], Optional
    parser.add_argument("--lora_weights", type=float, nargs="*")  # list[float], Optional
    parser.add_argument("--inject_timesteps", action="store_true")  # Default: False

    args = parser.parse_args()
    kwargs = vars(args)

    if omit_none:
        kwargs = {key: val for key, val in kwargs.items() if val is not None}

    return kwargs


async def main():
    kwargs = parse_kwargs()
    print(f"kwargs:\n{kwargs}")

    begin_prep = time.time()
    print("Preparing DMD2 pipeline and preprocessors...")

    pipe, preprocessors = await load_pipe_and_preprocessors(**kwargs)

    end_prep = time.time()
    print(f"Preparation time (pipeline + preprocessors): {end_prep - begin_prep:.2f} sec")
    print(f"Overall preparation time (imports + pipeline + preprocessors): {end_prep - begin_import:.2f} sec")

    gen_images = generate_images(pipe=pipe, preprocessors=preprocessors, **kwargs)

    save_images(gen_images)


if __name__ == "__main__":
    asyncio.run(main())
