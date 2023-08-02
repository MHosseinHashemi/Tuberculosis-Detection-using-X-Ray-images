## 1st Project: Tuberculosis Detection using X-Ray images 
This repository contains an implementation of a Tuberculosis (TB) detection project using TensorFlow. The goal of this project is to build a deep learning model that can automatically detect TB from chest X-ray images.

### Table of Contents
- Introduction
- Model Architecture
- Data
- Results
- Contributing
- License
- Acknowledgements

## Introduction
Tuberculosis is a contagious bacterial infection that primarily affects the lungs. Early detection of TB is crucial for effective treatment and prevention of the disease's spread. This project aims to leverage deep learning and computer vision techniques to automate the process of TB detection from chest X-ray images, thus aiding healthcare professionals in diagnosing the disease more efficiently and accurately.


## Model Architecture
The TB detection model is based on the EfficientNetV2S architecture, a powerful convolutional neural network (CNN) model. EfficientNetV2S has been pretrained on the ImageNet dataset, enabling it to learn rich and discriminative features from large-scale image data. The pretrained EfficientNetV2S model is fine-tuned using the provided chest X-ray images to adapt its features for TB detection.

## Data
The dataset used for this project consists of chest X-ray images with annotations indicating TB presence or absence. The dataset is divided into a training set with 5600 images and a validation set with 1400 images. The model is trained on the training set to learn the patterns associated with TB, and its performance is evaluated on the validation set to assess its generalization capabilities.

## Results
The model has been trained for 25 epochs and achieved the following results on the validation set:

- Validation Accuracy: 99.64%
- Validation Recall: 99.29%
- Validation Precision: 100.00%
- These high validation metrics indicate the model's ability to accurately identify TB in chest X-ray images.

## Contributing
Contributions to this project are welcome. If you wish to contribute, please follow the standard GitHub workflow by forking the repository and creating a pull request.

## License
This project is licensed under the MIT License. 

## Acknowledgements
We would like to express our gratitude to the authors of the EfficientNetV2S model and the creators of the dataset used in this project for providing valuable resources. Thank you to the open-source community for fostering collaboration and knowledge sharing.

