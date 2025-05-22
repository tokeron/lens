# Lens Environment Setup

This guide will help you set up the `lens` conda environment, which includes all the necessary dependencies, including a custom version of the `diffusers` library.

## Overview

This repository serves as the base for two research papers:
- **Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models**  
  This paper presents an in-depth analysis of the role and impact of padding tokens in text-to-image (T2I) diffusion models. It explores how padding tokens contribute to—or are ignored by—the image generation process.
- **Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models**  
  This work investigates how information flows across textual token representations in T2I models. It addresses challenges like semantic leakage and token redundancy, and proposes methods to mitigate these issues.

## Prerequisites

Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

## Creating the Environment

To create the `lens` environment, follow these steps:

Run the following command to create the environment:

```bash
conda env create -f lens_env.yaml
```

This command will create a new conda environment named lens and install all the required dependencies, including the custom diffusers library.

Activating the Environment

Once the environment is created, activate it using:

```bash
conda activate lens
```

Installing the Custom Diffusers Version

After activating the environment, install the custom version of diffusers by running:

```bash
pip install -e .
```

This ensures that your environment uses the modified version of the diffusers library required by this project.

Running the Lens Test

The repository includes a Lens Test as an example of how to run the custom diffuser wrapper. 
This test demonstrates how to use the following variables:


- **specific_token_idx_to_keep_per_prompt_lists**: A list of token IDs that should be kept during the generation process. Use this variable to specify which tokens from the prompt must be preserved.
- **merge_ranges**: Ranges indicating which tokens’ representations should be merged into a single representation. This helps in consolidating token information for improved generation results.
- **clean_tokens_ids**: A list of token IDs to be patched from a clean prompt. This variable defines which tokens in the current prompt should be replaced with those from a standardized, unaltered prompt.
- **clean_prompt**: The clean prompt that provides the baseline or unaltered token representations. This prompt is used to patch tokens as defined in the clean_tokens_ids variable.

For a detailed example, refer to the _lens_test_ script in the repository.

Additional Notes
- If you need to update the environment after making changes to the lens_env.yaml file, use:

```bash
conda env update -f lens_env.yaml
```

- To deactivate the environment, simply run:

```bash
conda deactivate
```

- If you encounter issues with installing the diffusers library, ensure your internet connection is active and verify the GitHub URL is correct. You may also need to upgrade pip:

```bash
pip install --upgrade pip
```


Citation

If you find this repository useful in your research, please cite the associated papers using the following BibTeX entries:

```latex
@misc{toker2025paddingtonemechanisticanalysis,
      title={Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models}, 
      author={Michael Toker and Ido Galil and Hadas Orgad and Rinon Gal and Yoad Tewel and Gal Chechik and Yonatan Belinkov},
      year={2025},
      eprint={2501.06751},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.06751}, 
}
```

```latex
@misc{kaplan2025followflowinformationflow,
      title={Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models}, 
      author={Guy Kaplan and Michael Toker and Yuval Reif and Yonatan Belinkov and Roy Schwartz},
      year={2025},
      eprint={2504.01137},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.01137}, 
}
```

Feel free to reach out if you encounter any issues or need further help!
