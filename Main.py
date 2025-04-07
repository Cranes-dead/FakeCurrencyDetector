# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
import tensorflow as tf
import joblib
import argparse

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
NUM_CLASSES = 2
RANDOM_STATE = 42
MODEL_PATH = "models/currency_classifier.joblib"
CNN_FEATURES_PATH = "models/cnn_features.joblib"

class FakeCurrencyDetector:
    def __init__(self, feature_type='cnn', classifier_type='svm', cnn_model='resnet50'):
        """
        Initialize the detector with specified feature extraction and classification methods
        
        Parameters:
        - feature_type: 'cnn', 'sift', or 'hybrid'
        - classifier_type: 'svm', 'rf', 'lr', or 'nn'
        - cnn_model: 'resnet50' or 'vgg16'
        """
        self.feature_type = feature_type
        self.classifier_type = classifier_type
        self.cnn_model_type = cnn_model
        self.classifier = None
        self.feature_extractor = None
        self.cnn_model = None
        self.scaler = StandardScaler()
        
        # Initialize feature extractor
        if feature_type in ['cnn', 'hybrid']:
            self._setup_cnn_model(cnn_model)
        
        # Initialize classifier
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        elif classifier_type == 'lr':
            self.classifier = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        elif classifier_type == 'nn':
            # Neural network will be set up during training
            pass
    
    def _setup_cnn_model(self, model_type='resnet50'):
        """Set up CNN model for feature extraction"""
        if model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        elif model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Create feature extraction model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        self.cnn_model = Model(inputs=base_model.input, outputs=x)
    
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
    
    def load_dataset(self, data_dir):
        """
        Load and prepare the dataset
        
        Returns:
        - features: array of features
        - labels: array of labels (0: Fake, 1: Real)
        - filenames: list of image filenames
        """
        features = []
        labels = []
        filenames = []
        
        # Classes mapping
        class_map = {'Fake': 0, 'Real': 1}
        
        # Process each class directory
        for class_name in ['Fake', 'Real']:
            class_dir = os.path.join(data_dir, class_name)
            class_label = class_map[class_name]
            
            for filename in os.listdir(class_dir):
                # Skip non-image files
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Load and process image
                img_path = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    feature = self.extract_features(img)
                    
                    features.append(feature)
                    labels.append(class_label)
                    filenames.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return np.array(features), np.array(labels), filenames
    
    def build_nn_classifier(self, input_shape):
        """Build a shallow neural network classifier"""
        model = tf.keras.Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir, val_dir=None, epochs=50):
        """Train the model"""
        # Load training data
        X_train, y_train, _ = self.load_dataset(train_dir)
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        
        # If validation directory is provided
        if val_dir:
            X_val, y_val, _ = self.load_dataset(val_dir)
            X_val = self.scaler.transform(X_val)
        
        # Train based on classifier type
        if self.classifier_type == 'nn':
            # Build neural network
            input_shape = X_train.shape[1]
            self.classifier = self.build_nn_classifier(input_shape)
            
            # Train with or without validation data
            if val_dir:
                self.classifier.fit(
                    X_train, y_train,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    verbose=1
                )
            else:
                self.classifier.fit(
                    X_train, y_train,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=1
                )
        else:
            # Train sklearn models
            self.classifier.fit(X_train, y_train)
    
    def evaluate(self, test_dir):
        """Evaluate the model on test data"""
        # Load test data
        X_test, y_test, filenames = self.load_dataset(test_dir)
        
        # Standardize features
        X_test = self.scaler.transform(X_test)
        
        # Get predictions
        if self.classifier_type == 'nn':
            y_pred = np.argmax(self.classifier.predict(X_test), axis=1)
            y_proba = self.classifier.predict(X_test)
        else:
            y_pred = self.classifier.predict(X_test)
            y_proba = self.classifier.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Return metrics and predictions
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
        
        return metrics, y_pred, y_proba, filenames
    
    def cross_validate(self, data_dir, n_splits=5):
        """Perform stratified k-fold cross-validation"""
        # Load all data
        X, y, _ = self.load_dataset(data_dir)
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        # Metrics lists
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # Cross-validation
        fold = 1
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            if self.classifier_type == 'nn':
                input_shape = X_train.shape[1]
                model = self.build_nn_classifier(input_shape)
                model.fit(X_train, y_train, epochs=30, verbose=0)
                y_pred = np.argmax(model.predict(X_test), axis=1)
            else:
                self.classifier.fit(X_train, y_train)
                y_pred = self.classifier.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store metrics
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
            
            print(f"Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            fold += 1
        
        # Print average metrics
        print("\nCross-Validation Results:")
        print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
        # Return average metrics
        return {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1_scores)
        }
    
    def predict(self, img_path):
        """Predict whether an image is of real or fake currency"""
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        features = self.extract_features(img)
        features = self.scaler.transform([features])
        
        # Get prediction
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
    
    def generate_gradcam(self, img_path, class_idx=None):
        """Generate Grad-CAM visualization to highlight important regions"""
        if self.feature_type != 'cnn' and self.feature_type != 'hybrid':
            print("Grad-CAM only available for CNN-based models")
            return None
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            result, _ = self.predict(img_path)
            class_idx = 1 if result == 'Real' else 0
        
        # Create Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=[self.cnn_model.inputs],
            outputs=[self.cnn_model.get_layer('conv5_block3_out' if self.cnn_model_type == 'resnet50' else 'block5_conv3').output, 
                    self.cnn_model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]
            
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
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
        
        return superimposed_img, heatmap
    
    def save_model(self, model_dir="models"):
        """Save the trained model and scaler"""
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save classifier
        if self.classifier_type == 'nn':
            self.classifier.save(os.path.join(model_dir, "nn_classifier.h5"))
        else:
            joblib.dump(self.classifier, os.path.join(model_dir, "classifier.joblib"))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        
        # Save feature extraction model if CNN-based
        if self.feature_type in ['cnn', 'hybrid']:
            self.cnn_model.save(os.path.join(model_dir, "cnn_feature_extractor.h5"))
    
    def load_model(self, model_dir="models"):
        """Load trained model and scaler"""
        # Load classifier
        if self.classifier_type == 'nn':
            self.classifier = tf.keras.models.load_model(os.path.join(model_dir, "nn_classifier.h5"))
        else:
            self.classifier = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        
        # Load feature extraction model if CNN-based
        if self.feature_type in ['cnn', 'hybrid']:
            self.cnn_model = tf.keras.models.load_model(os.path.join(model_dir, "cnn_feature_extractor.h5"))

def train_and_evaluate(feature_type='hybrid', classifier_type='svm', cnn_model='resnet50'):
    """Train and evaluate the model"""
    # Paths
    train_dir = "Dataset/Training"
    val_dir = "Dataset/Validation"
    test_dir = "Dataset/Testing"
    
    # Initialize detector
    detector = FakeCurrencyDetector(
        feature_type=feature_type, 
        classifier_type=classifier_type, 
        cnn_model=cnn_model
    )
    
    # Train
    print("Training model...")
    detector.train(train_dir, val_dir)
    
    # Cross-validate
    print("\nPerforming cross-validation...")
    cv_metrics = detector.cross_validate(train_dir)
    
    # Evaluate
    print("\nEvaluating on test data...")
    metrics, _, _, _ = detector.evaluate(test_dir)
    
    # Save model
    print("\nSaving model...")
    detector.save_model()
    
    return detector

# Streamlit app code moved to app.py

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fake Currency Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--feature-type', type=str, default='hybrid', 
                      choices=['cnn', 'sift', 'hybrid'], help='Feature extraction method')
    parser.add_argument('--classifier', type=str, default='svm',
                      choices=['svm', 'rf', 'lr', 'nn'], help='Classifier type')
    parser.add_argument('--cnn-model', type=str, default='resnet50',
                      choices=['resnet50', 'vgg16'], help='CNN model for feature extraction')
    
    args = parser.parse_args()
    
    if args.train:
        # Only run the training function when explicitly requested
        print("Training model with settings:")
        print(f"- Feature type: {args.feature_type}")
        print(f"- Classifier: {args.classifier}")
        print(f"- CNN model: {args.cnn_model}")
        
        train_and_evaluate(
            feature_type=args.feature_type,
            classifier_type=args.classifier,
            cnn_model=args.cnn_model
        )
    else:
        print("No action specified. Use --train to train the model or run 'streamlit run app.py' to launch the web interface.")
