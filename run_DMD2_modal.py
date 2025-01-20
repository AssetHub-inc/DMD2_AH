import shlex
import time

from PIL import Image
import modal

from t2i_adapter_4step import (
    load_DMD2_pipe,
    load_preprocessors,
    generate_images,
    parse_kwargs,
)

ARGS = """
  --t2i_adapter_names canny depth_zoe \
  --preprocessor_names canny depth_zoe \
  --adapter_conditioning_scale 1.0 0.6 \
  --prompt="Medieval knight armor, plain background, 4K HDR photo, high quality, masterpiece, extremely detailed, sharp, clear" \
  --negative_prompt="shadow, reflection, worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting" \
  --loras "add-detail-xl.safetensors" "Fantasy_Armors_XL.safetensors" \
  --lora_weights 3.0 1.0 \
  --image="inputs/color_armor.png" \
  --img2img \
  --strength 1.0 \
  --num_inference_steps 4 \
  --batchsize 8
"""

def get_args_from_string(args_str):
    return shlex.split(args_str)


class ImageGenInput():
    pass

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",  # OpenCV requirements
    )
    .pip_install(
        "torch",
        "torchvision",
        "xformers",
        index_url="https://download.pytorch.org/whl/cu124",  # CUDA 12.4
    )
    .pip_install(
        "accelerate",
        "controlnet_aux",
        "diffusers",
        "huggingface-hub[hf_transfer]",
        "mediapipe",
        "peft",
        "pillow",
        "transformers",
    )
    .workdir("/workspace/models/loras/")
    .run_commands(
        "apt install -y wget",  # TODO: move it into `.apt_install()` at the top
        "wget 'https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor' -O add-detail-xl.safetensors -q",
        "wget 'https://civitai.com/api/download/models/658549?type=Model&format=SafeTensor' -O Fantasy_Armors_XL.safetensors -q",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            # "HF_ENDPOINT": "https://hf-mirror.com",
            # "HF_HUB_CACHE_DIR": CACHE_DIR,
        }
    )
    
)

app = modal.App("DMD2", image=image)

@app.cls(
    gpu='a10g',
    cpu=2,
    memory=1024,
    timeout=3600,
    allow_concurrent_inputs=100,
    keep_warm=1,
)
class ImageGen:
    @modal.enter()
    async def enter(self):
        args = get_args_from_string(ARGS)
        self.kwargs = parse_kwargs(args=args)
        print(f"{self.kwargs = }")

        begin_prep = time.time()
        print("Preparing DMD2 pipeline and preprocessors...")

        self.pipe = load_DMD2_pipe(**self.kwargs)
        self.preprocessors = load_preprocessors(**self.kwargs)

        end_prep = time.time()
        print("Overall preparation time (imports + pipeline + preprocessors): "
              + f"{end_prep - begin_prep:.2f} sec")

    @modal.method()
    async def run(self, image: Image.Image):
        gen_images = generate_images(
            pipe=self.pipe,
            preprocessors=self.preprocessors,
            image=image,
            **self.kwargs
        )


if __name__ == "__main__":
    args = get_args_from_string(ARGS)
    kwargs = parse_kwargs(args=args)

    worker = ImageGen()
    worker.enter()
    worker.run()
