# Vision AI Intern Assignment

## Overview

This repository is my submission for the *Vision AI intern assignment* at Golden Owl.  
The goal is **image classification of cats vs. dogs**, using PyTorch.  
Also includes inference mode (single image or folder) and saves results in JSON.  
Accompanying this code is a short report describing the pipeline, challenges, and ideas for improvement.

## Environment & Requirements
- Python version: `3.8+` 

To install:
```bash
pip install -r requirements.txt
```
Or using Pipenv

```bash
pip install pipenv
pipenv install
pipenv shell
```

## Dataset
Images of cats and dogs from Kaggle: [Cat VS Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)

The dataset (images and JSON annotation files) can be downloaded [here](https://drive.google.com/file/d/1Fz_lRpogpOLHERh2znfiYy3Bg7ncKAoN/view?usp=sharing)

## Model & Pipeline
Model: A simple Convolutional Neural Network (CNN) defined in model.py
Model's configuration can be changed in [config\basic_cnn.yaml](config\basic_cnn.yaml)
Training setup:
- Loss: cross-entropy

- Optimizer: Adam with lr = 1e-3

- Epochs: 100

- Batch size: 32

Inference: Can run on a single image or folder of images.

Outputs:

- Predictions printed to console

- Results saved to inference_result.json with {filename: predicted_label} format.

## How to run
### Trainig
```bash
python train.py --config-file config/basic_cnn.yaml
```

### Inference
Single image:
```bash
python inference.py \
  --config-file config/basic_cnn.yaml \
  --image-file path/to/image.jpg \
  --checkpoint saved_models/best_model.pth

```

Folder of images:
```bash
python inference.py \
  --config-file config/basic_cnn.yaml \
  --image-file path/to/folder \
  --checkpoint saved_models/best_model.pth
```

## Report
Reports for both Image Classification and TTS tasks are available in the [`report/`](report/) folder.

## How to Reproduce
1.Clone this repo and install requirements
```bash
git clone https://github.com/DucAnh0Cap/goldenowl-assignment
pip install -r requirements
%cd goldenowl-assignment
```
2. Download [dataset](https://drive.google.com/file/d/1Fz_lRpogpOLHERh2znfiYy3Bg7ncKAoN/view?usp=sharing) and unzip

3. Train the model or use pretrained checkpoint
```bash
python train --config-file config/basic_cnn.yaml
```

4. Inference
```bash
python inference.py \
  --config-file config/basic_cnn.yaml \
  --image-file path/to/image.jpg \
  --checkpoint saved_models/best_model.pth
```

## Demo
To run demo on local machine
```bash
streamlit run app.py
```

## Notes
Ensure the checkpoint is placed under saved_models/

If using GPU, ensure CUDA version is compatible

