# Brain Tumor Detection and Classification Using Deep Learning

## Project Overview

Brain tumors are one of the most serious neurological disorders that require early and accurate diagnosis for effective treatment. Manual analysis of Magnetic Resonance Imaging (MRI) scans is time-consuming and depends on the expertise of radiologists. This project presents an AI-based brain tumor detection and classification system that automatically identifies the type of brain tumor from MRI images using Deep Learning models.

The application classifies brain MRI images into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The project compares the performance of multiple transfer learning models, including VGG16, ResNet50, and Xception, and deploys the best-performing model through a Streamlit web application.

---

## Objectives

- Detect brain tumors from MRI images.
- Classify MRI images into four categories.
- Compare the performance of multiple deep learning models.
- Improve diagnostic accuracy using transfer learning.
- Provide an easy-to-use web application for prediction.

---

## Dataset

The dataset consists of brain MRI images collected from Kaggle.

### Classes
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The dataset is divided into:
- Training Set
- Testing Set

---

## Data Preprocessing

The following preprocessing techniques were applied before training:

- Image resizing
- Pixel normalization
- Data augmentation
- Image labeling
- Batch generation

These preprocessing steps improve model performance and reduce overfitting.

---

## Deep Learning Models Used

The following pretrained CNN models were used:

- VGG16
- ResNet50
- Xception

Transfer learning was applied by using ImageNet pretrained weights and fine-tuning the final layers for brain tumor classification.

---

## Model Training

The models were trained using TensorFlow and Keras.

Training process included:

- Loading MRI images
- Preprocessing
- Feature extraction
- Model training
- Validation
- Performance evaluation

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score

---

## Best Model

Among the trained models, **Xception** achieved the highest overall performance on the test dataset and was selected for deployment.

---

## Project Workflow

1. Collect MRI dataset
2. Preprocess images
3. Apply data augmentation
4. Train deep learning models
5. Evaluate model performance
6. Save the best model
7. Deploy using Streamlit
8. Predict tumor type from uploaded MRI image

---

## Streamlit Application

The Streamlit application contains three sections:

### Home
- Project introduction
- Workflow
- Model overview

### Performance
- Model comparison
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Classification Report

### Prediction
- Upload MRI image
- Predict tumor class
- Display confidence score
- Display Grad-CAM heatmap for explainability

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL
- Streamlit

---

## Project Structure

```
Brain-Tumor-Detection/
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
├── models/
│   ├── VGG16.keras
│   ├── ResNet50.keras
│   └── Xception.keras
│
├── notebooks/
│   └── Brain_Tumor_Classification.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Brain-Tumor-Detection.git
```

### Navigate to project folder

```bash
cd Brain-Tumor-Detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Sample Prediction

Input:
- Upload an MRI brain scan image.

Output:
- Predicted Class
- Confidence Score
- Grad-CAM Visualization

Example Prediction:

```
Prediction : Glioma

Confidence : 98.4%
```

---

## Future Enhancements

- Support DICOM image format
- Deploy using cloud platforms
- Integrate with hospital systems
- Improve model accuracy using larger datasets
- Add segmentation before classification

---

## Conclusion

This project demonstrates the application of Deep Learning and Transfer Learning for automatic brain tumor detection and classification from MRI images. By comparing multiple CNN models and deploying the best-performing model through Streamlit, the system provides an efficient and user-friendly solution that can assist healthcare professionals in the early diagnosis of brain tumors.

---

## Author

**Syeda Afreen**

Bachelor of Engineering (Computer Science)

PES College of Engineering, Mandya
