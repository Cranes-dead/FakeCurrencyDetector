<h1 align="center">Fake Currency Detection System</h1>

<div align= "center"><img src="https://images.thequint.com/thequint%2F2016-11%2F71274674-012f-4a31-b1c8-a3ca4cbf4387%2Fnew-500-note-currency.jpg?rect=0%2C0%2C1400%2C788" width="350" height="187"/>
  <h4>A currency detection system which classifies currency as real or counterfiet using image processing.</h4>
</div>

<br>

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/viram-jain/FakeCurrencyDetectionSystem/issues)
[![Forks](https://img.shields.io/github/forks/viram-jain/FakeCurrencyDetectionSystem.svg?logo=github)](https://github.com/viram-jain/FakeCurrencyDetectionSystem/network/members)
[![Stargazers](https://img.shields.io/github/stars/viram-jain/FakeCurrencyDetectionSystem.svg?logo=github)](https://github.com/viram-jain/FakeCurrencyDetectionSystem/stargazers)
[![Issues](https://img.shields.io/github/issues/viram-jain/FakeCurrencyDetectionSystem.svg?logo=github)](https://github.com/viram-jain/FakeCurrencyDetectionSystem/issues)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://linkedin.com/in/viram-jain-43450018b)

## :innocent: Motivation
Recognition of fake Indian currency is very important in major domains like banking. This system is used to detect whether the currency is fake or original through the automated system which is through convolution neural networks, in deep learning. The problem is that common people these days are facing issues in regard to circulating fake currencies and are not able to recognize which notes are real and which are counterfeit. The main objective of this project is to make it convenient for any common man to know whether or not a note is real by using our desktop or android application.

## :warning: TechStack/Framework used

- [Python](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)

## :star: Features

- **Multiple Feature Extraction Techniques**:
  - SIFT (Scale-Invariant Feature Transform)
  - CNN-based feature extraction
  - Hybrid approach combining both

- **Advanced Data Augmentation**:
  - Rotations, flips, and shifts
  - Brightness and contrast adjustments
  - Zoom variations
  - Automatically multiplies small datasets

- **Multiple Classifier Options**:
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - Neural Network with regularization

- **Optimized for Small Datasets**:
  - MobileNetV2 support for efficient learning
  - L2 regularization to prevent overfitting
  - Increased dropout for better generalization

- **Visual Explanations**:
  - Grad-CAM visualizations for model decisions
  - SIFT keypoint visualization

## :bulb: Working

The program functions in the following way:
1. Images of real and fake currency are used to train the model
2. Data augmentation is applied to increase the effective dataset size
3. Features are extracted using selected methods (SIFT, CNN, or hybrid)
4. A classifier is trained on these features to distinguish real from fake currency
5. New images can be classified through the web interface or command line

## :rocket: Getting Started

### Prerequisites
- Python 3.6+
- Required packages: tensorflow, opencv-python, scikit-learn, streamlit

### Installation

```bash
# Clone the repository
git clone https://github.com/Cranes-dead/FakeCurrencyDetector.git

# Navigate to the project directory
cd FakeCurrencyDetectionSystem

# Install required packages
pip install -r requirements.txt
```

### Usage

#### Web Interface
```bash
# Run the Streamlit web app
streamlit run app.py
```

#### Command Line (Training)
```bash
# Train with default settings
python main.py --train

# Train with specific settings
python main.py --train --feature-type hybrid --classifier svm --cnn-model mobilenet --augment
```

## :wrench: Configuration Options

- **Feature Extraction Methods**:
  - `hybrid`: Combines CNN and SIFT features (recommended)
  - `cnn`: Uses only CNN features
  - `sift`: Uses only SIFT features

- **Classifier Types**:
  - `svm`: Support Vector Machine with RBF kernel
  - `rf`: Random Forest
  - `lr`: Logistic Regression
  - `nn`: Neural Network

- **CNN Models**:
  - `mobilenet`: MobileNetV2 (recommended for small datasets)
  - `resnet50`: ResNet50V2
  - `vgg16`: VGG16

## :heart: Owner
Made with :heart:&nbsp;  by [Viram Jain](https://github.com/Cranes-dead)

## :page_with_curl: License
This project is licensed under the MIT License - see the LICENSE file for details.
