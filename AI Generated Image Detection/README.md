# AI Generated Image Detection

This project predicts whether a given image is AI generated or not using BNExt-T as the backbone model along with various feature channels, and configuration options for fine-tuning model behavior during inference.


## Installation

1. Create a new Conda environment:
   ```bash
   conda create --name artifact-env python=3.10
   ```

3. Activate the Conda environment:
   ```bash
   conda activate artifact-env
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration (`model.cfg`)

The `model.cfg` file contains important configuration parameters.

### `dataset`:
- `path`: Specifies the path to the image folder.
- `labels`: The number of labels or categories in the dataset (e.g., 2 for binary classification).
- `name`: The name of the dataset being used.

### `model`:
- `add_fft_channel`: If set to `true`, adds the FFT (Fast Fourier Transform) channel to the model.
- `add_lbp_channel`: If set to `true`, adds the LBP (Local Binary Pattern) channel to the model.
- `add_magnitude_channel`: If set to `false`, prevents the addition of the magnitude channel to the model.
- `backbone`: The backbone model to use for feature extraction, e.g., `BNext-T`.
- `freeze_backbone`: If set to `false`, allows the backbone model to be fine-tuned during training.

### `inference`:
- `batch_size`: The batch size to use during inference (default is `32`).
- `limit_test_batches`: The fraction of test batches to use (default is `1.0`, meaning use all test batches).
- `mixed_precision`: If `true`, enables mixed-precision training/inference for faster processing.
- `resolution`: The resolution to which input images will be resized during inference (default is `224`).
- `seed`: The random seed for initialization and reproducibility (default is `5`).

## Usage

### Running the Model

To run the model, use the `main.py` file. Make sure to set the dataset path in the `model.cfg` file.

```bash
python main.py --cfg <path_to_model_cfg>
```

This command will load the configuration from the `model.cfg` file and begin the model inference process.

### Command-Line Arguments:
- `--cfg`: Path to the `model.cfg` configuration file. This file contains important settings for the dataset and model.

### Example:
```bash
python main.py --cfg ./model.cfg
```