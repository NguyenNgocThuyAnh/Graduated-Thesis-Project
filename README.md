# **EXPLORING TRANSFER LEARNING IN DEEP NEURAL NETWORKS FOR ENHANCED MEDICAL IMAGE CLASSIFICATION: APPLICATIONS IN LEUKEMIA DIAGNOSIS**

## **1. Overview of the Project**

This project investigates the application of transfer learning with deep Convolutional Neural Networks (CNNs) for the automated classification of blood cells in leukemia diagnosis. Using the publicly available Blood Cell Images for Cancer Detection dataset, the study implemented and compared multiple CNN architectures, including VGG16, ResNet50, Xception, DenseNet201, and EfficientNetB3, on the proposed architecture to indentify the most efficient and interpretable model for real-world medical applications.

The project presents these main insights:

* The application of transfer learning to enhance performance in medical image classification with limited data.

* A comparative analysis of five CNN architectures in terms of accuracy, efficiency, and computational cost.

* The integration of explainable AI techniques (Grad-CAM) to visualize and interpret model predictions.

* Model training, fine-tuning, and evaluation conducted on Google Colab GPU environment for efficient experimentation.

* The potential of transfer learning–based CNNs to support AI-assisted leukemia diagnosis and improve clinical decision-making.

## **2. Research Objectives**

* Improve diagnostic accuracy and efficiency for leukemia detection from microscopic blood cell images.

* Apply transfer learning to fine-tune pre-trained CNN models for a five-class blood cell classification task.
  
* Evaluate model performance comprehensively using Accuracy, F1-score, ROC-AUC, and Grad-CAM visualizations.
  
* Compare performance, resource efficiency, and explainability of the selected models.
  
* Propose future research directions for each model generalization and clinical deployment.

## **3. Research Methodology**

  ### Dataset
  * Source: [Blood Cell Images for Cancer Detection](https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection)
  
  * Size: 5,000 high-resolution microscopic images
  
  * Image Size: 1024x1024 pixels minimum
  
  * Staining: Wright-Giemsa Magnification: 100x oil immersion (1000x total)
  
  * Color: 24-bit RGB Multiple focal planes per sample
  
  * Classes: Basophil, Erythroblast, Monocyte, Myeloblast, Segmented Neutrophil
  
  * License: Attribution 4.0 International (CC BY 4.0)
  
  * DOI Citation: https://doi.org/10.34740/kaggle/dsv/10500753

  ### Workflow

  * Data Preprocessing:
  
    * Image resizing

    * Applied each model’s built-in preprocess_input() function (from TensorFlow/Keras) to preprocess all images according to the model’s native format.
  
  * Fine-tuning to Identify Optimal Architecture: Applied the transfer learning technique to fine-tuning and build an optimal architecture for 5 implemented CNN models.
  
  * Model Training: Adjusted hyperparameters and trained models basing on the proposed architecture.
  
  * Evaluation: Computed metrics and visualize results with Grad-CAM.

  * Comparison: Analyzed models' performance (accuracy, F1-score, AUC, and computational cost) and proposed enhancing strategies.

  ### Experimental Environment

  * Language: Python

  * Frameworks: TensorFlow, Keras

  * Environment: Google Colab (GPU runtime)
    
  * Libraries: See requirements.txt
  
  * For optinmal training progress, GPU acceleration is strongly recommended.

## 4. Instructions for Implementation

  ### Step 1: Setup Google Colab Environment
  * Open a new Google Colab notebook.
    
  * In the top menu, go to Runtime → Change runtime type → Hardware accelerator → Choose T4 GPU option.

  * Upload the dataset folder to the Google Colab workspace and mount Google Drive.

  * For example:
   
    ```python
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    ```
  ### Step 2: Install Necessary Libraries
  * Run ```import ...``` or ```from ... import ...``` to import needed libraries.

  * For example:
     ```python
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import pickle
    import seaborn as sns
    
    from tensorflow import keras
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from keras.applications import vgg16
    from keras.callbacks import EarlyStopping
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    ```
  ### Step 3: Run the Model Notebook
  * Open the notebook of the model you want to run (e.g. DenseNet201.ipynb).

  * Execute all cells sequentially.

  * The notebook performs:

    * Data Checking - verify the data path and randomly display images from each category.
    
    * Data Preprocessing – resize images and apply each model’s built-in preprocess_input() function.
    
    * Model Training – fine-tune pre-trained CNN models using transfer learning on GPU.

    ``` python
    history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping])
    ```
    
    * Evaluation – compute accuracy, F1-score, confusion matrix, ROC–AUC, and accuracy - loss plots.
    
    * Explainability – visualize Grad-CAM heatmaps for AI interpretability.
   
  ### Step 4: Save and Load Trained Models
  * After training, model weights and training histories are automatically saved with the following structure:
  ``` python
  Saved Models/
      ├── vgg16.keras
      ├── resnet50.keras
      ├── xception.keras
      ├── densenet201.keras
      ├── efficientnetb3.keras
      ├── vgg16_training_history.pkl
      ├── resnet50_training_history.pkl
      ├── xception_training_history.pkl
      ├── densenet201_training_history.pkl
      └── efficientnetb3_training_history.pkl
  ```
  * To reload a model for inference or evaluation, use the ```load_model()``` function available in Tensorflow library.

  * For example:

  ``` python
  # Load the saved model
  model = tf.keras.models.load_model('/content/drive/MyDrive/Tổng hợp Đồ án Tốt nghiệp/Saved Models/vgg16.keras')
  ```
  ### Step 5: Visualize Results
  * Run the visualization section at the end of each notebook to display:

    * Training accuracy and loss curves

    * Confusion matrices and ROC curves

    * Grad-CAM heatmaps highlighting areas most influential to predictions
   
  * All outputs appear directly inside Colab for inspection.

