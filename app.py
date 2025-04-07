import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16
import joblib
from PIL import Image
import io
import time

# Load model and components
class FakeCurrencyDetector:
    def __init__(self, model_dir="models", feature_type='hybrid', classifier_type='svm', cnn_model='resnet50'):
        """Initialize the detector with pre-trained models"""
        self.feature_type = feature_type
        self.classifier_type = classifier_type
        self.cnn_model_type = cnn_model
        self.classifier = None
        self.cnn_model = None
        self.scaler = None
        
        # Load the components
        self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """Load all model components"""
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load classifier
        try:
            if self.classifier_type == 'nn':
                self.classifier = tf.keras.models.load_model(os.path.join(model_dir, "nn_classifier.h5"))
            else:
                self.classifier = joblib.load(os.path.join(model_dir, "classifier.joblib"))
                
            # Load scaler
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            
            # Load CNN model
            if self.feature_type in ['cnn', 'hybrid']:
                try:
                    self.cnn_model = tf.keras.models.load_model(os.path.join(model_dir, "cnn_feature_extractor.h5"))
                except:
                    # If the saved model isn't found, create a new one
                    self._setup_cnn_model(self.cnn_model_type)
        except FileNotFoundError:
            # If model files don't exist, set up basic functionality
            st.error("Model files not found. Please train the model first.")
            if self.feature_type in ['cnn', 'hybrid']:
                self._setup_cnn_model(self.cnn_model_type)
    
    def _setup_cnn_model(self, model_type='resnet50'):
        """Set up CNN model for feature extraction"""
        IMG_HEIGHT, IMG_WIDTH = 224, 224
        
        if model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        elif model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Create feature extraction model
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.cnn_model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    def _extract_sift_features(self, img):
        """Extract SIFT features from an image"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # If no keypoints found, return zero vector
        if descriptors is None or len(keypoints) == 0:
            return np.zeros(128)
        
        # Return mean of descriptors to have fixed-size feature vector
        return np.mean(descriptors, axis=0)
    
    def _extract_cnn_features(self, img):
        """Extract features using CNN"""
        # Resize and preprocess
        IMG_HEIGHT, IMG_WIDTH = 224, 224
        img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess based on the model
        if self.cnn_model_type == 'resnet50':
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        elif self.cnn_model_type == 'vgg16':
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            
        # Extract features
        features = self.cnn_model.predict(img_array)
        return features[0]
    
    def extract_features(self, img):
        """Extract features based on the selected method"""
        if self.feature_type == 'sift':
            return self._extract_sift_features(img)
        elif self.feature_type == 'cnn':
            return self._extract_cnn_features(img)
        elif self.feature_type == 'hybrid':
            sift_features = self._extract_sift_features(img)
            cnn_features = self._extract_cnn_features(img)
            return np.concatenate([sift_features, cnn_features])
    
    def predict(self, img):
        """Predict whether an image is of real or fake currency"""
        # Extract features
        features = self.extract_features(img)
        
        # If scaler is available, standardize features
        if self.scaler is not None:
            features = self.scaler.transform([features])
        else:
            features = features.reshape(1, -1)
        
        # Get prediction
        if self.classifier is None:
            return "Model not loaded", 0.0
        
        try:
            if self.classifier_type == 'nn':
                proba = self.classifier.predict(features)[0]
                pred_class = np.argmax(proba)
                confidence = proba[pred_class]
            else:
                pred_class = self.classifier.predict(features)[0]
                confidence = np.max(self.classifier.predict_proba(features)[0])
            
            # Return prediction and confidence
            result = 'Real' if pred_class == 1 else 'Fake'
            return result, confidence
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def generate_gradcam(self, img, class_idx=None):
        """Generate Grad-CAM visualization to highlight important regions"""
        if self.feature_type != 'cnn' and self.feature_type != 'hybrid':
            return None, "Grad-CAM only available for CNN-based models"
        
        if self.cnn_model is None:
            return None, "CNN model not loaded"
        
        # Prepare image
        IMG_HEIGHT, IMG_WIDTH = 224, 224
        img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img_tensor = image.img_to_array(img_resized)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # Preprocess based on the model
        if self.cnn_model_type == 'resnet50':
            img_tensor = tf.keras.applications.resnet50.preprocess_input(img_tensor)
        elif self.cnn_model_type == 'vgg16':
            img_tensor = tf.keras.applications.vgg16.preprocess_input(img_tensor)
        
        # Determine prediction class if not provided
        if class_idx is None:
            result, _ = self.predict(img)
            class_idx = 1 if result == 'Real' else 0
        
        try:
            # Get the last convolutional layer
            if self.cnn_model_type == 'resnet50':
                last_conv_layer = 'conv5_block3_out'
            else:  # VGG16
                last_conv_layer = 'block5_conv3'
            
            # Create Grad-CAM model
            grad_model = tf.keras.models.Model(
                inputs=[self.cnn_model.inputs],
                outputs=[self.cnn_model.get_layer(last_conv_layer).output, self.cnn_model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                class_channel = predictions[:, class_idx]
                
            # Extract gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Apply pooling to gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps with gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to original image size
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Convert heatmap to RGB
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose heatmap on original image
            superimposed_img = heatmap * 0.4 + img
            superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
            
            return superimposed_img, None
        except Exception as e:
            return None, f"Error generating Grad-CAM: {str(e)}"
    
    def get_sift_keypoints(self, img):
        """Get SIFT keypoints visualization"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints
        keypoints, _ = sift.detectAndCompute(gray, None)
        
        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(
            gray, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return img_with_keypoints, keypoints

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Fake Currency Detector",
        page_icon="💵",
        layout="wide"
    )
    
    st.title("Fake Currency Detection System")
    st.markdown("""
    This application uses computer vision techniques to determine if a currency note is real or counterfeit.
    Upload an image of a currency note to get started.
    """)
    
    # Sidebar for model configuration
    st.sidebar.title("Model Configuration")
    
    feature_type = st.sidebar.selectbox(
        "Feature Extraction Method",
        ["hybrid", "cnn", "sift"],
        format_func=lambda x: {
            "cnn": "CNN Features Only",
            "sift": "SIFT Features Only",
            "hybrid": "Hybrid (CNN + SIFT) [Recommended]"
        }.get(x, x)
    )
    
    classifier_type = st.sidebar.selectbox(
        "Classifier",
        ["svm", "rf", "lr", "nn"],
        format_func=lambda x: {
            "svm": "Support Vector Machine (RBF Kernel)",
            "rf": "Random Forest",
            "lr": "Logistic Regression",
            "nn": "Neural Network"
        }.get(x, x)
    )
    
    cnn_model = "resnet50"
    if feature_type in ["cnn", "hybrid"]:
        cnn_model = st.sidebar.selectbox(
            "CNN Architecture",
            ["resnet50", "vgg16"],
            format_func=lambda x: {
                "resnet50": "ResNet50",
                "vgg16": "VGG16"
            }.get(x, x)
        )
    
    # Initialize detector
    detector = FakeCurrencyDetector(
        feature_type=feature_type,
        classifier_type=classifier_type,
        cnn_model=cnn_model
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image of currency", type=["jpg", "jpeg", "png"])
    
    # Camera input
    camera_input = st.camera_input("Or take a photo")
    
    # Process the image
    image_file = uploaded_file if uploaded_file is not None else camera_input
    
    if image_file is not None:
        # Read image
        image_bytes = image_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case it's RGBA)
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array for processing
        img = np.array(pil_image)
        
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            # Make prediction
            start_time = time.time()
            result, confidence = detector.predict(img)
            end_time = time.time()
            
            # Generate visualizations
            if feature_type in ['cnn', 'hybrid']:
                grad_cam_img, error_msg = detector.generate_gradcam(img)
            
            if feature_type in ['sift', 'hybrid']:
                sift_img, keypoints = detector.get_sift_keypoints(img)
        
        with col2:
            # Display prediction result
            result_color = "green" if result == "Real" else "red"
            st.markdown(f"<h2 style='color: {result_color};'>Prediction: {result} Currency</h2>", unsafe_allow_html=True)
            
            # Display confidence with progress bar
            st.write(f"Confidence: {confidence*100:.2f}%")
            st.progress(float(confidence))
            
            # Display processing time
            st.write(f"Processing time: {end_time - start_time:.2f} seconds")
        
        # Display visualizations
        st.subheader("Visualizations")
        vis_cols = st.columns(2)
        
        # Show Grad-CAM if available
        if feature_type in ['cnn', 'hybrid'] and grad_cam_img is not None:
            with vis_cols[0]:
                st.subheader("Grad-CAM Visualization")
                st.image(grad_cam_img, caption="Regions influencing prediction", use_column_width=True)
                st.write("Hotter colors (red) indicate areas that strongly influenced the model's decision.")
        elif feature_type in ['cnn', 'hybrid'] and error_msg:
            with vis_cols[0]:
                st.warning(error_msg)
        
        # Show SIFT keypoints if available
        if feature_type in ['sift', 'hybrid']:
            with vis_cols[1]:
                st.subheader("SIFT Keypoints")
                st.image(sift_img, caption=f"Detected {len(keypoints)} keypoints", use_column_width=True)
                st.write("SIFT keypoints show distinctive local features used for authentication.")
        
        # Add explanation based on the detection
        st.subheader("Analysis")
        if result == "Fake":
            st.warning("This currency appears to be counterfeit. The model has detected irregularities in the image patterns, colors, or security features.")
            
            # List possible issues (for demonstration)
            st.write("Possible issues detected:")
            issues = [
                "Inconsistent printing patterns",
                "Anomalies in security features",
                "Color irregularities",
                "Texture inconsistencies"
            ]
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("This currency appears to be genuine. The model has verified the expected patterns, colors, and security features.")
        
        # Add disclaimer
        st.info("Disclaimer: This tool is for educational purposes only. For official currency verification, please consult with banking authorities.")

    # Add information about the model
    with st.sidebar.expander("About the Model"):
        st.write("""
        This system uses both traditional computer vision techniques (SIFT) and deep learning (CNN) to detect counterfeit currency.
        
        **Features used:**
        - SIFT (Scale-Invariant Feature Transform): Detects local features independent of scale and rotation
        - CNN (Convolutional Neural Network): Extracts high-level features from the image
        
        **Classifiers available:**
        - SVM with RBF kernel: Good for non-linear classification
        - Random Forest: Ensemble method resistant to overfitting
        - Logistic Regression: Simple linear classifier
        - Neural Network: Deep learning classifier
        """)

if __name__ == "__main__":
    main()