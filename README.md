# Depthy

This repository contains the scripts for training and testing the depth estimation model:

1. **`train.py`**: For training the depth estimation model.
2. **`test.py`**: For testing the model on a single image and generating a depth map.

---

## Requirements

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- Matplotlib
- gdown (for downloading datasets)

You can install the required packages using:

### 1. Training the Model (train.py)

The train.py script is used to train the depth estimation model on a dataset. It supports the following features:

1) Loading pretrained weights for the model.
2) Loading a checkpoint to resume training.
3) Configurable loss weights for the combined loss function.
4) Adjustable number of training epochs.

**Parameters**

--pretrained: Use pretrained weights for the model. If this flag is not provided, the model will be initialized randomly.
--checkpoint (optional): Path to a checkpoint file to resume training. If not specified, training starts from scratch (or with pretrained weights, if --pretrained is used).
--loss_weights: Weights for the combined loss function. Provide three values:
SSIM weight: Weight for the structural similarity index loss.
BerHu weight: Weight for the BerHu loss.
Gradient weight: Weight for the gradient-based loss.
Default: 0.5 0.5 0.
--epochs: Number of training epochs. Default: 10.

Example Commands:

Train with Pretrained Weights (No Checkpoint):

python train.py --pretrained --loss_weights 0.5 0.5 0 --epochs 5

Resume Training from a Checkpoint:

python train.py --checkpoint best_model2.pth --loss_weights 0.5 0.3 0.2 --epochs 5

Train Without Pretrained Weights:

python train.py --loss_weights 0.6 0.4 0 --epochs 5

### 2. Testing the Model (test.py)
The test.py script is used to test the trained model on a single image and visualize the depth map.

Parameters
--pretrained: Use pretrained weights for the model. If not provided, the model will use the weights from the checkpoint only.
--checkpoint: Path to the model checkpoint file. This parameter is required to load the trained model.
--image_path: Path to the input image for testing. This parameter is required.
Example Commands

Test with Pretrained Weights and Checkpoint:

python test.py --pretrained --checkpoint best_model2.pth --image_path /path/to/image.jpg

Test with Checkpoint Only:

python test.py --checkpoint best_model2.pth --image_path /path/to/image.jpg

**Training** (train.py)
During training, the script will:

- Download and extract the dataset (if not already downloaded).
- Train the model and output the following:
- Training losses (SSIM, BerHu, Gradient, and Total) for each epoch.
- Validation losses for each epoch.
- Save the best-performing model to best_model2.pth in the current directory.
- Automatically stop training early if validation loss does not improve for a specified number of epochs (early stopping).

**Testing** (test.py)
During testing, the script will:

- Load the input image and preprocess it.
- Generate the depth map using the trained model.
- Display the depth map using Matplotlib with the plasma colormap.

## Notes
1. Ensure your checkpoint file exists and is accessible when using the --checkpoint parameter.
2. The dataset will be automatically downloaded and extracted when running train.py. Ensure you have enough disk space.
3. Use GPU for faster training and testing if available. The scripts automatically detect GPU availability.
