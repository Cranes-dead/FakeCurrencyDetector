# Fake Currency Detection System

This project implements an advanced fake currency detection system using computer vision techniques and machine learning. It combines traditional computer vision methods like SIFT with deep learning models to provide accurate detection of counterfeit currency.

## Features

- **Multiple Feature Extraction Methods**:
  - CNN features from pre-trained models (ResNet50, VGG16)
  - SIFT (Scale-Invariant Feature Transform) features
  - Hybrid approach combining both CNN and SIFT features

- **Multiple Classifier Options**:
  - Support Vector Machine with RBF kernel
  - Random Forest
  - Logistic Regression
  - Shallow neural network

- **Advanced Visualization**:
  - Grad-CAM visualization highlighting regions influencing the prediction
  - SIFT keypoint visualization
  - Confidence scores and metrics

- **Robust Evaluation**:
  - Stratified K-fold cross-validation
  - Comprehensive metrics: Accuracy, Precision, Recall, F1, Confusion Matrix

- **User-Friendly Interface**:
  - Streamlit web application for easy use
  - Support for image upload or camera capture
  - Real-time analysis and feedback

## Directory Structure

```
└── FakeCurrencyDetectionSystem
    ├── Dataset
    │   ├── Testing
    │   │   ├── Fake.jpeg
    │   │   └── Real.jpg
    │   ├── Training
    │   │   ├── Fake
    │   │   └── Real
    │   └── Validation
    │       ├── Fake
    │       └── Real
    ├── models
    │   ├── classifier.joblib
    │   ├── cnn_feature_extractor.h5
    │   ├── nn_classifier.h5 (optional)
    │   └── scaler.joblib
    ├── main.py
    └── app.py
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- Streamlit
- Matplotlib
- NumPy
- joblib

You can install all required packages with:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-currency-detection.git
   cd fake-currency-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the dataset organized as shown in the directory structure.

## Usage

### Training the Model

To train the model with the default settings (hybrid features, SVM classifier):

```bash
python main.py --train
```

For custom configuration:

```bash
python main.py --train --feature-type hybrid --classifier svm --cnn-model resnet50
```

### Running the Web App

```bash
streamlit run app.py
```

This will launch a local web server and open the app in your default browser.

## How It Works

### 1. Feature Extraction

The system offers three methods for feature extraction:

- **CNN Features**: Uses pre-trained models (ResNet50 or VGG16) to extract high-level features from currency images.
- **SIFT Features**: Extracts scale and rotation-invariant local features that capture distinctive patterns and textures.
- **Hybrid Features**: Combines both CNN and SIFT features for more robust detection.

### 2. Classification

Various classifiers are implemented to provide flexibility and performance:

- **SVM with RBF kernel**: Effective for non-linear classification.
- **Random Forest**: Ensemble method that works well with small datasets.
- **Logistic Regression**: Simple linear classifier that can perform well with good features.
- **Neural Network**: A shallow network that can learn complex patterns.

### 3. Visualization

- **Grad-CAM**: Highlights regions of the currency that influenced the model's decision.
- **SIFT Keypoints**: Shows the distinctive local features used for classification.

### 4. Evaluation

The system performs:
- **Stratified K-fold Cross-validation**: To ensure reliable performance assessment with the small dataset.
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix.

## Extending the Project

### Adding More Features

You can extend the feature extraction methods:
- Add more CNN architectures (MobileNet, EfficientNet, etc.)
- Implement other feature extraction methods (HOG, LBP, etc.)
- Add color histogram analysis

### Enhancing the Web App

- Add batch processing
- Implement model comparison
- Add model retraining capability through the UI
- Include explanations of specific counterfeit detection markers

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original project was based on a ResNet50 implementation for currency detection
- Thanks to the OpenCV and TensorFlow communities for their excellent tools
