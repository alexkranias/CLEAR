import torch
import os
import gc
import requests
from safetensors.torch import load_file
from diffusers import FluxPipeline
from pipeline_flux_img2img import FluxImg2ImgPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import LocalFlexAttnProcessor, LocalDownsampleFlexAttnProcessor, init_local_mask_flex, init_local_downsample_mask_flex


from torch.profiler import profile, record_function, ProfilerActivity

bfl_repo="black-forest-labs/FLUX.1-dev"
device = torch.device('cuda')
dtype = torch.bfloat16
pipe = FluxPipeline.from_pretrained(bfl_repo, torch_dtype=dtype).to(device)


prompt = "enchanted forest, glowing plants, towering ancient trees, a mystical girl, magical aura, " \
         "fantasy style, vibrant colors, ethereal lighting, bokeh effect, ultra-detailed, painterly, ultra HD, 8K, " \
         "soft glowing lights, mist and fog, otherworldly ambiance, glowing mushrooms, sparkling particles"
height = 512
width = 1024
image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=3.5,
    num_inference_steps=20,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image


height = 2048
width = 4096
down_factor, window_size = 4, 8
# Supported Configurations:
# down_factor, window_size = 1, 8
# down_factor, window_size = 1, 16
# down_factor, window_size = 1, 32
# down_factor, window_size = 4, 16
# down_factor, window_size = 4, 8
if down_factor == 1:
    init_local_mask_flex(height // 16, width // 16, text_length=512, window_size=window_size, device=device)
    attn_processors = {}
    for k in pipe.transformer.attn_processors.keys():
        attn_processors[k] = LocalFlexAttnProcessor()
else:
    init_local_downsample_mask_flex(height // 16, width // 16, text_length=512, window_size=window_size, down_factor=down_factor, device=device)
    attn_processors = {}
    for k in pipe.transformer.attn_processors.keys():
        attn_processors[k] = LocalDownsampleFlexAttnProcessor(down_factor=down_factor).to(device, dtype)
pipe.transformer.set_attn_processor(attn_processors)


if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if down_factor == 1:
    if not os.path.exists(f'ckpt/clear_local_{window_size}.safetensors'):
        print(f'Checkpoint not found. Downloading checkpoint to ckpt/clear_local_{window_size}.safetensors')
        response = requests.get(f"https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_{window_size}.safetensors")
        response.raise_for_status()
        with open(f'ckpt/clear_local_{window_size}.safetensors', 'wb') as f:
            f.write(response.content)
    state_dict = load_file(f'ckpt/clear_local_{window_size}.safetensors')
else:
    if not os.path.exists(f'ckpt/clear_local_{window_size}_down_{down_factor}.safetensors'):
        print(f'Checkpoint not found. Downloading checkpoint to ckpt/clear_local_{window_size}_down_{down_factor}.safetensors')
        response = requests.get(f"https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_{window_size}_down_{down_factor}.safetensors")
        response.raise_for_status()
        with open(f'ckpt/clear_local_{window_size}_down_{down_factor}.safetensors', 'wb') as f:
            f.write(response.content)
    state_dict = load_file(f'ckpt/clear_local_{window_size}_down_{down_factor}.safetensors')

missing_keys, unexpected_keys = pipe.transformer.load_state_dict(state_dict, strict=False)

missing_keys = list(filter(lambda p: ('.attn.to_q.' in p or 
                                      '.attn.to_k.' in p or 
                                      '.attn.to_v.' in p or 
                                      '.attn.to_out.' in p or 
                                      'spatial_weight' in p), missing_keys))


if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    print(
        f"Loading attn weights from state_dict led to unexpected keys: {unexpected_keys}"
        f" and missing keys: {missing_keys}."
    )

strength = 0.7

# pytorch profiler
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        image_hr = pipe(prompt=prompt,
                        image=image.resize((width, height)),
                        strength=strength,
                        num_inference_steps=20, 
                        guidance_scale=7.5, 
                        height=height,
                        width=width,
                        ntk_factor=10,
                        proportional_attention=True,
                        generator=torch.Generator("cpu").manual_seed(0)
                        ).images[0]
        image_hr

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))