# Brain Tumor MRI Classification Using Deep Learning

A deep learning project for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The project explores data understanding, image preprocessing, data augmentation, CNN-based modeling, transfer learning, evaluation, and a simple Flask prediction API.

## Project Overview

Brain MRI analysis can be time-consuming and requires specialized expertise. This project investigates an automated image-classification pipeline using TensorFlow/Keras and transfer-learning architectures.

The notebook includes experiments with:

- CNN
- VGG16
- MobileNetV2
- ResNet50
- Image preprocessing and normalization
- Data augmentation
- Validation and test evaluation
- Confusion matrices and classification reports
- Single-image prediction
- Flask-based prediction API

## Dataset

The project uses the **Brain Tumor MRI Dataset** from Kaggle by Masoud Nickparvar.

Dataset size:

- **7,033 grayscale MRI images**
- **5,712 training images**
- **1,321 testing images**

Classes:

`glioma`, `meningioma`, `pituitary`, `notumor`

> The dataset is not included in this repository. Download it from Kaggle and place it locally according to the paths used in the notebook.

## Preprocessing

The notebook performs image preprocessing including:

- Grayscale image loading
- Brain-contour cropping
- Resizing
- Pixel normalization
- Conversion to 3-channel images for compatibility with ImageNet-pretrained architectures
- Data augmentation for training

The processed images are organized into class-specific directories for TensorFlow/Keras data generators.

## Modeling

### VGG16

The main final experiment uses transfer learning with **VGG16** pretrained on ImageNet.

Key configuration in the notebook:

- Input size: `124 × 124 × 3`
- Image augmentation
- Frozen and fine-tuned VGG16 layers
- Global Average Pooling
- Dense layers with L2 regularization
- Batch normalization
- Dropout
- Adam optimizer
- Learning rate: `1e-5`
- Early stopping
- Learning-rate reduction

### Other Experiments

The notebook also contains experiments using:

- Custom CNN
- MobileNetV2
- ResNet50

These experiments are included to document the modeling process and comparison work.

## Results

The final VGG16 experiment recorded the following test-set results in the original notebook:

| Metric | Result |
|---|---:|
| Test Accuracy | **98.40%** |
| Weighted Precision | **98.41%** |
| Weighted Recall | **98.40%** |
| Weighted F1 Score | **98.39%** |

These figures are the results recorded in the notebook and should be interpreted in the context of this specific dataset and experimental setup.

## Repository Structure

```text
brain-tumor-mri-classification/
│
├── brain_tumor_mri_classification.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Notebook

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Brain Tumor MRI Dataset from Kaggle and place the extracted dataset where the notebook expects it, or update the dataset paths in the notebook.

### 5. Run the notebook

The project was originally developed in Google Colab. You can upload the notebook to Google Colab and run the cells there, or adapt the `/content/...` paths for local execution.

## Flask API

The repository also includes a simplified Flask API in `app.py`.

Place the trained model file:

```text
my_vgg_model.h5
```

in the project directory, then run:

```bash
python app.py
```

The API exposes:

```text
POST /predict
```

with an image uploaded under the `file` form field.

Example response:

```json
{
  "predicted_class": "glioma",
  "confidence": 0.98,
  "probabilities": {
    "glioma": 0.98,
    "meningioma": 0.01,
    "notumor": 0.00,
    "pituitary": 0.01
  }
}
```

## Important Note

This project is an academic machine-learning project for image classification and experimentation. It is **not a medical diagnostic system** and should not be used to make clinical decisions.

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- Jupyter / Google Colab

## Author

**Nadia Omar Khair**

Data Science & Artificial Intelligence Graduate

