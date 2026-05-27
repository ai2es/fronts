# NOAA FrontFinder v2

This repository is the updated version of NOAA's FrontFinder code, created by Andrew Justin (@andrewjustin) and updated by Taylor Mandelbaum (@aaTman). 

## Key updates:

- **New YAML-based configuration**
    - Single source of truth
    - A fully documented approach that avoids needing to manage shell scripts
    - YAMLs can be version controlled and tied to commits

- **Fully-typed code**
    - All code in the repository is now typed and checked via `pyrefly`

- **Formatted code**
    - Code is linted and formatted using `ruff`

- **Updated Tensorflow**
    - Tensorflow is updated as is all ML code
    - Updating TF prevents issues with deprecated or unoptimized code

- **Managed environments using `pixi`**
    - FrontFinder is trained on the OU supercomputer, schooner. Schooner has some unique requirements, such as being locked into `glibc <= 2.17`. `pixi` enables sequestered unique virtual environments (features) along with lockfiles to manage dependencies. 
    - Allows distinct virtual environments for specific tasks, such as training, prediction, development, etc.
    - Can port environments to non-pixi-friendly resources using `pixi-pack` if needed
    - Can handle both conda and PyPI dependencies in one `pyproject.toml` (or `pixi.toml`) file
    - **Much** faster than conda

## Getting Started

### Installing

Clone this repository:

```git clone https://github.com/ai2es/fronts.git```

Enter the newly cloned directory and install the default environment using `pixi`:

```
cd fronts
pixi install
```

## Training

The model is available as a `.keras` model which [is much easier](https://www.tensorflow.org/tutorials/keras/save_and_load#new_high-level_keras_format) to manage, debug, etc. If you want to train it from scratch, you need the data.

### Generating Data

Downloading all of the data for the model, which includes ERA5 and satellite data, is straightforward but will take a long time:

```pixi run -e data generate_data --icechunk_path \path\to\your\icechunk_store```

This downloads all of the data and stores it in an icechunk store specified in the YAML or given by `--icechunk_path`. This YAML is default, located at `src/fronts/configs/generate_data.yaml`. Note that there is currently a default `icechunk_path` that will create the store at the root of this repository. 

If you want to invoke your own YAML with all of the required attributes:

```pixi run -e data generate_data --config /path/to/your.yaml```

The fronts truth data from 2018-2024 is available on request.

### Training the Model

Assuming you have all the data in the icechunk store:

```pixi run -e train train_model```

This uses `src/fronts/configs/train_model.yaml` to train a UNet3+.

## Prediction

Predicting using the `.keras` model is straightforward as well:

```pixi run -e predict generate_predictions```

Where the YAML will predict the hold-out year for this project, 2019, for the domain 0.25N to 80N, 130E to 9.75E (i.e. 130-369.75). You can specify what to predict and where the data is located within your own YAML, either with a `start_date`, `end_date` and `frequency`, or `dates` in ISO 8601 formatting. You can specify either or both.