import os
import sys

import argparse
from tqdm.auto import tqdm
from PIL import Image

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from transformers import LlamaForCausalLM, LlamaTokenizer


from huggingface_hub import hf_hub_download
from argparse import ArgumentParser
from box import Box
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF

import torch

from tqdm import tqdm
from PIL import Image
import torch
from PIL import Image
import requests
from scipy.spatial.distance import cosine
import time


class TextToImage:
    def __init__(self, model_name, ckpt_dir, num_images=1, device="cuda", seed=None):
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.num_images = num_images
        self.seed=seed
        self.load_model_components()

    def load_model_components(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    def forward(self, prompt, num_images=1, per_token=False, per_range=False):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_complementory_range(self, range_to_keep, max_length):
        return list(set(range(max_length)) - set(range_to_keep))

    def get_ranges_single_tokenizer(self, prompt, tokenizer, max_length):
        ranges = {}
        tokens = tokenizer(prompt, return_tensors="pt")['input_ids'][0]
        token_length = len(tokens)
        decoded_tokens = [tokenizer.decode(token.item()) for token in tokens]
        pad_length = max_length - token_length

        ranges['full'] = []
        # ranges['tokens'] = list(range(0, token_length))
        # ranges['pads'] = list(range(token_length, max_length))
        # ranges['eot'] = list(range(0, len(tokens)-1)) + list(range(len(tokens), max_length))
        # ranges['None'] = list(range(0, max_length))

        return ranges
    
    
    def get_ranges_all_tokenizers(self, prompt, tokenizers, max_lengths):
        ranges = None
        for (tokenizer_key, tokenizer), max_len in zip(tokenizers.items(), max_lengths):
            print(f"Getting ranges for {tokenizer_key}")
            updated_ranges = self.get_ranges_single_tokenizer(prompt, tokenizer, max_len)
            if ranges is None:
                ranges = {key: [value] for key, value in updated_ranges.items()}
            else:
                for key, value in updated_ranges.items():
                    if key in ranges:
                        ranges[key].append(value)
                    else:
                        ranges[key] = [value]

        # Ensure each key has exactly len(tokenizers) items in its list
        bad_keys = []
        for key in ranges:
            if len(ranges[key]) < len(tokenizers):
                bad_keys.append(key)
            # while len(ranges[key]) < len(tokenizers):
                # ranges[key].append([range(0, 77)])  # Append a full mask range if not present
        for key in bad_keys:
            print(f"Removing key {key} from ranges")
            ranges.pop(key)
                
        return ranges
    
    def create_image_grid(self, images, output_path=None, number_of_images_per_row=2, do_save=False):
        print(f'Creating image grid for {len(images)} images')
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths[:number_of_images_per_row])  # width of 10 images
        total_height = sum(heights[i] for i in range(0, len(heights), number_of_images_per_row))  # height of every 10th image

        new_img = Image.new('RGB', (total_width, total_height))

        x_offset = 0
        y_offset = 0
        for i, img in enumerate(images):
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width
            if (i + 1) % number_of_images_per_row == 0:  # move to next row after every 10 images
                x_offset = 0
                y_offset += img.height
        if do_save:
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            new_img.save(output_path)
            print(f'Image grid saved to {output_path}')
        else:
            return new_img


class StableDiffusion3TextToImage(TextToImage):
    def __init__(self, model_name, ckpt_dir, num_images, device="cuda", seed=42, max_sequence_length=256):
        super().__init__(model_name, ckpt_dir, num_images, device, seed)
        self.max_sequence_length = max_sequence_length

    def load_model_components(self):
        # from diffusers_local.src.diffusers import StableDiffusion3Pipeline
        from diffusers import StableDiffusion3Pipeline
        generator = torch.manual_seed(self.seed)
        self.generator = generator
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        self.pipe = pipe.to(self.device)

    def forward(self, prompt, num_images, output_path, pad_str=None, save_grid=False, 
                save_per_image=True, zero_paddings=False, replace_with_pads=False, turn_attention_off=False):
        tokenizer = self.pipe.tokenizer
        tokenizer_2 = self.pipe.tokenizer_2
        tokenizer_3 = self.pipe.tokenizer_3
        tokenizers = {
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
            'tokenizer_3': tokenizer_3
        }
        number_of_tokens = self.pipe.tokenizer(prompt, return_tensors="pt")['input_ids'][0].shape[0]
        original_max_sequence_length = self.max_sequence_length
        max_sequence_length = self.max_sequence_length
        if max_sequence_length is None:
            max_sequence_length = 256
        elif max_sequence_length == 'prompt_len':
            max_sequence_length = number_of_tokens
        elif max_sequence_length == '2prompt_len':
            max_sequence_length = 2 * number_of_tokens

        ranges_to_try = self.get_ranges_all_tokenizers(prompt=prompt, tokenizers=tokenizers, max_lengths=[max_sequence_length] * 3)
        for skip_tokens_name, skip_tokens in ranges_to_try.items():
            prompt_words = prompt.split()
            prompt_snippet = "_".join(prompt_words[:15] if len(prompt_words) >= 5 else prompt_words)
            prompt_path = os.path.join(output_path, prompt_snippet)
            if not os.path.exists(prompt_path):
                os.makedirs(prompt_path)
            # check if exists before contine
            if save_grid and os.path.exists(os.path.join(prompt_path, f"{skip_tokens_name}.png")):
                print(f"Image grid for {skip_tokens_name} already exists, skipping")
                continue
            if save_per_image and os.path.exists(os.path.join(prompt_path, f"{skip_tokens_name}_0.png")):
                print(f"Image for {skip_tokens_name} already exists, skipping")
                continue

            pipe_output = self.pipe(prompt, num_images_per_prompt=num_images, generator=self.generator, skip_tokens=skip_tokens,
                               num_inference_steps=50, pad_encoders=None, replace_with_pads=replace_with_pads,
                                turn_attention_off=turn_attention_off, max_sequence_length=max_sequence_length)
            
            images = pipe_output.images

            if save_grid:
                grid_output_path = os.path.join(prompt_path, f"{skip_tokens_name}.png")
                self.create_image_grid(images=images, output_path=grid_output_path, 
                                        number_of_images_per_row=(self.num_images + 1) // 2, do_save=True)
                print(f"Saved the image grid to path: {grid_output_path}")

            if save_per_image:
                for i, image in enumerate(images):
                    curr_image_path = os.path.join(prompt_path, f"{skip_tokens_name}_{i}.png")
                    image.save(curr_image_path)
                print(f"Saved the image to path: {curr_image_path}")
                print(f"Generated image for {skip_tokens_name} to {output_path}")
                

class FluxTextToImage(TextToImage):
    def __init__(self, model_name, ckpt_dir, num_images, device="cuda", max_sequence_length=512, seed=42, mask_diffusion=False):
        super().__init__(model_name, ckpt_dir, num_images, device)
        self.max_sequence_length = 512 if max_sequence_length == None else max_sequence_length
        self.seed = seed
        self.mask_diffusion = mask_diffusion
    def load_model_components(self):
        # Implement loading of Flux model components
        from diffusers import FluxPipeline
        if self.model_name == "flux-schnell":
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        elif self.model_name == "flux-dev":
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Model name {self.model_name} not recognized")
    

        self.pipe = pipe.to(self.device)


    def forward(self, prompt, num_images, output_path, pad_str=None, save_grid=False, 
                save_per_image=True, zero_paddings=False, replace_with_pads=False, 
                turn_attention_off=False, num_inference_steps=4 ):
        tokenizer = self.pipe.tokenizer
        tokenizer_2 = self.pipe.tokenizer_2
        tokenizers = {
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
        }
        original_max_sequence_length = self.max_sequence_length
        prompt_ids_len = len(tokenizer_2(prompt)['input_ids'])
        if self.max_sequence_length == 'prompt_len':
            max_sequence_length = prompt_ids_len
        elif self.max_sequence_length == '2prompt_len':
            max_sequence_length = 2 * prompt_ids_len
        else:
            max_sequence_length = self.max_sequence_length
        ranges_to_try = self.get_ranges_all_tokenizers(prompt=prompt, tokenizers=tokenizers, max_lengths=[max_sequence_length] * 2)
        for skip_tokens_name, skip_tokens in ranges_to_try.items():
            prompt_words = prompt.split()
            prompt_snippet = "_".join(prompt_words[:15] if len(prompt_words) >= 5 else prompt_words)
            prompt_path = os.path.join(output_path, prompt_snippet)
            if not os.path.exists(prompt_path):
                os.makedirs(prompt_path)
                
            prompt_input_ids_len = len(tokenizer_2(prompt)['input_ids'])
            if self.mask_diffusion:
                prompt_copy = prompt
                prompt = [prompt_copy, '', prompt_copy, '']
                num_images = 1

            if self.model_name == "flux-dev":
                images = self.pipe(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=max_sequence_length,
                    generator=torch.Generator("cpu").manual_seed(self.seed),
                    clip_skip=None,
                    skip_tokens=skip_tokens,
                    num_images_per_prompt=num_images,
                    prompt_len=prompt_input_ids_len,
                    mask_diffusion=self.mask_diffusion
                ).images

            elif self.model_name == "flux-schnell":
                images = self.pipe(
                    prompt=prompt,
                    guidance_scale=0.,
                    height=512,
                    width=512,
                    max_sequence_length=max_sequence_length,
                    generator=torch.Generator("cpu").manual_seed(self.seed),
                    clip_skip=None,
                    skip_tokens=skip_tokens,
                    num_images_per_prompt=num_images,
                ).images
            else:
                raise ValueError(f"Model name {self.model_name} not recognized")
            
            os.makedirs(prompt_path, exist_ok=True)
            if save_grid:
                grid_output_path = os.path.join(prompt_path, f"{skip_tokens_name}.png")
                self.create_image_grid(images=images, output_path=grid_output_path, 
                                        number_of_images_per_row=(self.num_images + 1) // 2, do_save=True)
                print(f"Saved the image grid to path: {grid_output_path}")
            if save_per_image:
                for i, image in enumerate(images):
                    curr_image_path = os.path.join(prompt_path, f"{skip_tokens_name}_{i}.png")
                    image.save(curr_image_path)
                print(f"Saved the image to path: {curr_image_path}")
                print(f"Generated image for {skip_tokens_name} to {output_path}")

class StableDiffusion2TextToImage(TextToImage):
    def __init__(self, model_name, ckpt_dir, num_images, device="cuda", seed=42,):
        super().__init__(model_name, ckpt_dir, num_images, device, seed)
    def load_model_components(self):
        from diffusers import StableDiffusionPipeline
        generator = torch.manual_seed(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1',
            # variant="fp16",
            # repo_type="huggingface",
            torch_dtype=torch.float32,
            # use_safetensors=True,
            generator=generator,
        )
        pipe.to(device)

        self.pipe = pipe
        self.generator = generator
        self.device = device

    def forward(self, prompt, num_images, output_path, pad_str=None, save_grid=False, 
                save_per_image=True, zero_paddings=False, replace_with_pads=False, turn_attention_off=False):
        tokenizer = self.pipe.tokenizer

        ranges_to_try = self.get_ranges_single_tokenizer(prompt=prompt, tokenizer=tokenizer, max_length=77)
        for skip_tokens_name, skip_tokens in ranges_to_try.items():
            prompt_words = prompt.split()
            prompt_snippet = "_".join(prompt_words[:15] if len(prompt_words) >= 5 else prompt_words)
            prompt_path = os.path.join(output_path, prompt_snippet)
            if not os.path.exists(prompt_path):
                os.makedirs(prompt_path)
            # check if exists before contine
            if save_grid and os.path.exists(os.path.join(prompt_path, f"{skip_tokens_name}.png")):
                print(f"Image grid for {skip_tokens_name} already exists, skipping")
                continue
            if save_per_image and os.path.exists(os.path.join(prompt_path, f"{skip_tokens_name}_0.png")):
                print(f"Image for {skip_tokens_name} already exists, skipping")
                continue

            pipe_output = self.pipe(prompt, num_images_per_prompt=num_images, generator=self.generator, skip_tokens=skip_tokens,
                               num_inference_steps=20) #, pad_encoders=None, replace_with_pads=replace_with_pads, turn_attention_off=turn_attention_off)
            
            images = pipe_output.images
            if save_grid:
                grid_output_path = os.path.join(prompt_path, f"{skip_tokens_name}.png")
                self.create_image_grid(images=images, output_path=grid_output_path, 
                                        number_of_images_per_row=(self.num_images + 1) // 2, do_save=True)
                print(f"Saved the image grid to path: {grid_output_path}")
            if save_per_image:
                for i, image in enumerate(images):
                    curr_image_path = os.path.join(prompt_path, f"{skip_tokens_name}_{i}.png")
                    image.save(curr_image_path)
                print(f"Saved the image to path: {curr_image_path}")
                print(f"Generated image for {skip_tokens_name} to {output_path}")

class StableDiffusionXLPipelineTextToImage(TextToImage):
    def __init__(self, model_name, ckpt_dir, num_images, device="cuda", seed=42, max_sequence_length=77):
        super().__init__(model_name, ckpt_dir, num_images, device, seed)
        self.max_sequence_length = max_sequence_length

    def load_model_components(self):
        from diffusers import StableDiffusionXLPipeline
        generator = torch.manual_seed(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            # repo_type="huggingface",
            torch_dtype=torch.float16,
            # use_safetensors=True,
            generator=generator,
        )
        pipe.to(device)

        self.pipe = pipe
        self.generator = generator
        self.device = device

    def forward(self, prompt, num_images, output_path, pad_str=None, save_grid=False,
                save_per_image=True, zero_paddings=False, replace_with_pads=False, turn_attention_off=False):
        tokenizer = self.pipe.tokenizer
        tokenizer_2 = self.pipe.tokenizer_2
        tokenizers = {
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
        }
        ranges_to_try = self.get_ranges_all_tokenizers(prompt=prompt, tokenizers=tokenizers, max_lengths=[77] * 2)
        for skip_tokens_name, skip_tokens in ranges_to_try.items():
            prompt_words = prompt.split()
            prompt_snippet = "_".join(prompt_words[:15] if len(prompt_words) >= 5 else prompt_words)
            prompt_path = os.path.join(output_path, prompt_snippet)
            if not os.path.exists(prompt_path):
                os.makedirs(prompt_path)
            # check if exists before contine
            if save_grid and os.path.exists(os.path.join(prompt_path, f"{skip_tokens_name}.png")):
                print(f"Image grid for {skip_tokens_name} already exists, skipping")
                continue
            if save_per_image and os.path.exists(os.path.join(prompt_path, f"{skip_tokens_name}_0.png")):
                print(f"Image for {skip_tokens_name} already exists, skipping")
                continue
            max_sequence_length = self.max_sequence_length
            if max_sequence_length == None:
                max_sequence_length = 77
            elif max_sequence_length == 'prompt_len':
                max_sequence_length = len(tokenizer(prompt)['input_ids'])
            elif max_sequence_length == '2prompt_len':
                max_sequence_length = 2 * len(tokenizer(prompt)['input_ids'])
            else:
                print(f"max_sequence_length: {max_sequence_length}")
            with torch.no_grad():
                import time
                num_tokens = len(self.pipe.tokenizer_2(prompt)['input_ids'])
                max_sequence_length = num_tokens
                start_time = time.time()
                pipe_output = self.pipe(prompt, num_images_per_prompt=num_images, generator=self.generator, skip_tokens=skip_tokens,
                                   num_inference_steps=50, pad_encoders=None, max_sequence_length=max_sequence_length)
                end_time = time.time()
                print(f"Time taken for inference: {end_time - start_time}")
                
            images = pipe_output.images
            if save_grid:
                grid_output_path = os.path.join(prompt_path, f"{skip_tokens_name}.png")
                self.create_image_grid(images=images, output_path=grid_output_path, 
                                        number_of_images_per_row=(self.num_images + 1) // 2, do_save=True)
                print(f"Saved the image grid to path: {grid_output_path}")
            if save_per_image:
                for i, image in enumerate(images):
                    curr_image_path = os.path.join(prompt_path, f"{skip_tokens_name}_{i}.png")
                    image.save(curr_image_path)
                print(f"Saved the image to path: {curr_image_path}")
                print(f"Generated image for {skip_tokens_name} to {output_path}")

