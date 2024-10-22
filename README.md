# Lens Environment Setup

This guide will help you set up the `lens` conda environment, which includes all the necessary dependencies, including a custom version of the `diffusers` library.

## Prerequisites

Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

## Creating the Environment

To create the `lens` environment, follow these steps:

Run the following command to create the environment:

    ```bash
    conda env create -f lens_env.yaml
    ```

    This command will create a new conda environment named `lens` and install all the required dependencies, including the custom `diffusers` library.

## Activating the Environment

Once the environment is created, activate it using:

```bash
conda activate lens
```

## Additional Notes

- If you need to update the environment after making changes to the `lens_env.yaml` file, use the command:

    ```bash
    conda env update -f lens_env.yaml
    ```

- To deactivate the environment, simply use:

    ```bash
    conda deactivate
    ```

## Troubleshooting

- If you run into issues with installing the `diffusers` library, ensure that you have an active internet connection and that the GitHub URL is correct.
- You may also need to upgrade `pip` within the environment:

    ```bash
    pip install --upgrade pip
    ```

Feel free to reach out if you encounter any issues or need further help!

