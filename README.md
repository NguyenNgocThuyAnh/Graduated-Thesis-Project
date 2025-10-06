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

