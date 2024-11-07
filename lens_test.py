from diffusers_wrapper import StableDiffusion3TextToImage, FluxTextToImage, StableDiffusion2TextToImage, StableDiffusionXLPipelineTextToImage
import os

num_images = 2
main_output_path = '/home/tok/diffusers/lens/output'
model_name = 'flux-schnell' # 'sd2' or 'sd3' or 'sdxl' or 'flux-dev' or 'flux-schnell'
max_sequence_lengths ={
    'sd2': 77,
    'sd3': 77,
    'sdxl': 256,
    'flux-dev': 512,
    'flux-schnell': 256
}
max_sequence_length = max_sequence_lengths[model_name]


if 'flux' in model_name:
    ckpt_dir = ''
    model_class = FluxTextToImage(model_name=model_name, ckpt_dir=ckpt_dir, num_images=num_images,
                                  max_sequence_length=max_sequence_length)
elif model_name == 'sd2':
    model_name = model_name
    ckpt_dir = ''
    model_class = StableDiffusion2TextToImage(model_name=model_name, ckpt_dir=ckpt_dir, num_images=num_images)
elif model_name == 'sd3':
    model_name = model_name
    ckpt_dir = ''
    model_class = StableDiffusion3TextToImage(model_name=model_name, ckpt_dir=ckpt_dir, num_images=num_images,
                                              max_sequence_length=max_sequence_length)
elif model_name == 'sdxl':
    model_name = model_name
    ckpt_dir = ''
    model_class = StableDiffusionXLPipelineTextToImage(model_name=model_name, ckpt_dir=ckpt_dir, num_images=num_images,
                                                       max_sequence_length=max_sequence_length)
else:
    raise ValueError("Model name not found")


skip_layers = 0
return_grids = True
ranges_to_keep = ['full', 'tokens', 'specific_tokens']
prompts = [
    'kids playing in the playground',
]
specific_tokens_per_prompt = [
    ['playground', 'playing']
]

# Inference
for prompt, specific_tokens in zip(prompts, specific_tokens_per_prompt):
    print("generating images for prompt:", prompt)
    output_path = f'{main_output_path}/{model_name}/{prompt}'
    os.makedirs(output_path, exist_ok=True)
    grids = model_class.forward(prompt, num_images=model_class.num_images,
                        output_path=output_path,
                        save_grid=True,
                        save_per_image=False,
                        return_grids=return_grids,
                        skip_layers=skip_layers,
                        ranges_to_keep=ranges_to_keep,
                        specific_tokens=specific_tokens
                        # zero_paddings=False,
                        # replace_with_pads=True,
                        # turn_attention_off=False,
                        )
    # print("done generating images for prompt:", prompt)
    # for grid in grids:
    #     grid.show()