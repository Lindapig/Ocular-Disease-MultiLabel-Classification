# Ocular Disease Recognition using InceptionV3

## Introduction

Early diagnosis and treatment of ophthalmic diseases such as glaucoma, cataract, and AMD are crucial as they significantly impact quality of life.  
Eye fundus images play a vital role in identifying these ophthalmologic conditions.  
This project focuses on building an ocular disease recognition model using eye fundus images to train multi-label classifications for ocular diseases in a supervised learning framework.  
Model performance was evaluated using metrics such as area under the curve (AUC) of receiver operating characteristics (ROC) and other evaluation techniques.

## Dataset

The `ODIR-5K` dataset comprises eye fundus images from over 3,000 patients for training.  
Each pair of left and right fundus images is annotated with 8 labels, including normal, 5 major ocular diseases, and two other disease categories.

## Model

- Utilized **InceptionV3** as the base model for transfer learning.
- Custom layers were added for multi-label classification.
- Fine-tuned the model to optimize performance on the dataset.

## Training and Evaluation

- **Training Parameters**:
  - Optimizer: Adam with a learning rate of 0.0001.
  - Loss Function: Binary cross-entropy.
  - Batch Size: 32.
  - Epochs: 50.

- **Evaluation Metrics**:
  - **ROC-AUC**: Assessed classification reliability.
  - **Confusion Matrices**: Evaluated multi-label predictions.
  - **F1-Score**: Measured balance between precision and recall.
  - Accuracy was tested item-wise and label-wise for comprehensive validation.

## Results

- The model demonstrated high accuracy in classifying normal vs. abnormal cases and identifying specific ocular diseases.
- Achieved robust performance across all metrics and highlighted the potential of transfer learning in medical image classification.

## Future Work

- Expand the dataset with more diverse cases and additional disease categories.
- Explore lightweight models for real-time clinical deployment.
- Incorporate explainable AI techniques, such as Grad-CAM, to improve interpretability.

## Requirements

- Python 3.8 or later
- TensorFlow 2.x
- Keras
- OpenCV
- Seaborn
- Matplotlib
- scikit-learn

## Usage

Run main.py:

chmod +x run.sh

./run.sh
