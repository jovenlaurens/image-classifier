# STAT4060J Final Project: Unsupervised Image Clustering on Digital Handwritten Digits

## Introduction

This projet is an unsupervised learning of image classification utilizing the MNIST dataset. The goal is to cluster the images into 10 groups, each representing a digit from 0 to 9. The model is built using PyTorch and trained on Google Colab. The model is then tested on a local machine.

## Pre-requisites

For the requirment you can install the packages by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

or you can install the packages manually

## Data Set

In case you are checking where is the data set. I am using MNIST Data Set, where I got it from Tensorflow Library

## Usage

There are two files available for running the code:

- model_test.py
- model_test.ipynb

You can choose either file to execute the code. `model_test.py` is a Python script, and `model_test.ipynb` is a Jupyter notebook.

Personally, I recommend using `model_test.py` to run the code due to a known issue with PyTorch in Jupyter notebooks.

When running the file, make sure to modify two variables:

- `train`: Indicates whether the model is training or loading a pre-trained model.
- `type`: Specifies the number of epochs for training.
