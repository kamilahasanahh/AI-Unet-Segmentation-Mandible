# Mandible Segmentation using U-Net

This repository contains a PyTorch-based implementation for automating the segmentation of the mandible from medical images (e.g., panoramic X-rays or CT slices). The project utilizes a U-Net architecture with a pre-trained encoder to achieve high-accuracy segmentation.

## 🚀 Key Features

* **Model Architecture:** Implements a U-Net model using the `segmentation-models-pytorch` library, featuring a **ResNet-34** encoder pre-trained on ImageNet.
* **Performance Metrics:** Evaluates model performance using **Dice Score**, **Intersection over Union (IoU)**, and **Pixel Accuracy**.
* **Custom Data Pipeline:** Includes a specialized `MandibleDataset` class for handling grayscale images and binary masks, including automatic resizing (900x400) and padding to ensure dimensions are multiples of 32 for the U-Net architecture.
* **Visualization:** Integrated plotting functions to monitor training/validation loss and metric progression over epochs.

## 🛠️ Requirements

The following libraries are required to run the notebook:

* `torch` & `torchvision`
* `segmentation-models-pytorch`
* `numpy`
* `opencv-python` (cv2)
* `matplotlib`
* `tqdm`

## 📊 Dataset Structure

The notebook is configured to load data from Google Drive in the following structure:

```text
/MyDrive/segmentasi/train/mandible/
├── images/  # Input grayscale images
└── masks/   # Ground truth binary segmentation masks

```

## ⚙️ Training Configuration

* **Loss Function:** Dice Loss (configured for multiclass/binary segmentation).
* **Optimizer:** RMSprop with a learning rate of `1e-5`.
* **Epochs:** 100
* **Batch Size:** 1
* **Validation Split:** 20% of the training dataset.

## 📈 Results

The model tracks several metrics throughout training. In the initial phases, the notebook recorded the following for Epoch 1:

* **Train Dice Score:** ~0.48
* **Validation Dice Score:** ~0.56
* **Train Accuracy:** ~98%

Best model weights are automatically saved to `checkpoints/unet_model_best.pth` based on the lowest validation loss.

## 📝 Usage

1. **Mount Google Drive:** The notebook starts by mounting Drive to access the dataset.
2. **Install Dependencies:** Run the `!pip install segmentation-models-pytorch` cell.
3. **Run Training:** Execute the main training block. Training checkpoints will be saved in a local `/checkpoints` directory.
