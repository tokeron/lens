from diffusers_wrapper import StableDiffusion3TextToImage, FluxTextToImage, StableDiffusion2TextToImage, StableDiffusionXLPipelineTextToImage
from diffusers_wrapper import StableDiffusionAttendAndExciteTextToImage, StableDiffusionTextToImage
import os
import pandas as pd

def main():
    path_to_csv = '/home/tok/diffusers/lens/items_for_interpretability_table copy.csv'
    df = pd.read_csv(path_to_csv)
    extended_df = pd.DataFrame(columns=['Category', 'Subcategory', 'Examples', 'Correct Words'])

    # Iterate over the rows
    for index, row in df.iterrows():
        category = row['Category']
        subcategory = row['Subcategory']
        examples = row['Examples']
        if category == 'Typos':
            examples_with_correct_word = examples.split(',')
            examples = [example.split('(')[0].strip().replace("\"", "") for example in examples_with_correct_word]
            correct_words = [example.split('(')[1].split(')')[0].strip() for example in examples_with_correct_word]
            for i in range(len(examples)):
                extended_df = pd.concat([extended_df, pd.DataFrame({'Category': category, 'Subcategory': subcategory, 'Examples': examples[i], 'Correct Words': correct_words[i]}, index=[0])], ignore_index=True)
        else:
            examples = examples.split(',')
            examples = [example.strip() for example in examples]
            for example in examples:
                extended_df = pd.concat([extended_df, pd.DataFrame({'Category': category, 'Subcategory': subcategory, 'Examples': example, 'Correct Words': ''}, index=[0])], ignore_index=True)
    # print(extended_df)

    model_name = 'sdxl' # flux-schnell' # 'sd2' or 'sd3' or 'sdxl' or 'flux-dev' or 'flux-schnell'
    num_images = 1
    max_sequence_lengths ={
        'sd_aae': 77,
        'sd': 77,
        'sd2': 77,
        'sd3': 77,
        'sdxl': 77,
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
    elif model_name == 'sd_aae':
        ckpt_dir = ''
        model_class = StableDiffusionAttendAndExciteTextToImage(model_name=model_name, ckpt_dir=ckpt_dir, num_images=num_images,
                                                   max_sequence_length=max_sequence_length)
    elif model_name == 'sd':
        ckpt_dir = ''
        model_class = StableDiffusionTextToImage(model_name=model_name, ckpt_dir=ckpt_dir, num_images=num_images,
                                                 max_sequence_length=max_sequence_length
                                                 )
        
    else:
        raise ValueError("Model name not found")

    for index, row in extended_df.iterrows():
        # for skip_layers in range(0, 24):
        example = row['Examples']
        correct_word = row['Correct Words']
        category = row['Category']
        subcategory = row['Subcategory']
        main_output_path = f'/home/tok/diffusers/lens/output/{category}_{subcategory}'
        return_grids = True
        ranges_to_keep = ['specific_token_idx_to_keep_per_prompt', 'full'] # 'specific_tokens', # 'full', 'tokens'
        prompts = [example]
        if 'flux' in model_name:
            tokenizer = model_class.get_tokenizers()['tokenizer_2']
        else:
            tokenizer = model_class.get_tokenizers()['tokenizer']
        # print each token in the prompt
        tokens = tokenizer(example)
        specific_token_idx_to_keep_per_prompt_lists = []
        for idx, token in enumerate(tokens.input_ids):
            print(f'{idx}:{tokenizer.decode([token])}')
            specific_token_idx_to_keep_per_prompt_lists.append([idx])

        # specific_token_idx_to_keep_per_prompt_lists = [specific_token_idx_to_keep_per_prompt_lists[-1]]
        specific_tokens_per_prompt = [[]]
        merge_ranges = [
            [2,3],
            [4,5]
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
                                # skip_layers=skip_layers,
                                ranges_to_keep=ranges_to_keep,
                                specific_tokens=specific_tokens,
                                specific_token_idx_to_keep_per_prompt_lists=specific_token_idx_to_keep_per_prompt_lists,
                                merge_ranges=merge_ranges
                                # zero_paddings=False,
                                # replace_with_pads=True,
                                # turn_attention_off=False,
                                )
            # print("done generating images for prompt:", prompt)
            # for grid in grids:
            #     grid.show()


# define main
if __name__ == '__main__':
    main()