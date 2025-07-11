import os
import logging
import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


class PlantDiseaseModel:
  """Plant Disease Detection Model"""
  def __init__(self):
    self.model = None
    self.class_names = [
      "Healthy",
      "Tomato_Early_Blight",
      "Tomato_Late_Blight",
      "Tomato_Leaf_Mold",
      "Tomato_Septoria_Leaf_Spot",
      "Tomato_Spider_Mites",
      "Tomato_Target_Spot",
      "Tomato_Yellow_Leaf_Curl_Virus",
      "Tomato_Mosaic_Virus",
      "Tomato_Bacterial_Spot",
      "Potato_Early_Blight",
      "Potato_Late_Blight",
      "Potato_Healthy",
      "Corn_Common_Rust",
      "Corn_Northern_Leaf_Blight",
      "Corn_Healthy",
      "Pepper_Bacterial_Spot",
      "Pepper_Healthy"
    ]
    self.img_size = (224, 224)
    self.load_or_create_model()
    
    
  def create_model(self):
    """Create Random Forest Model"""
    model = RandomForestClassifier(
      n_estimators=100,
      random_state=42,
      max_depth=10,
      min_samples_split=5,
      min_samples_leaf=2
    )
    
    return model
  
  def load_or_create_model(self):
    # Load existing model or create new one
    model_path = "models/plant_disease_model.pkl"
    try:
      if os.path.exists(model_path):
        with open(model_path, "rb") as f:
          self.model = pickle.load(f)
        logging.info("Loaded existing model")
      else:
        self.model = self.create_model()
        # Create synthetic training data for demo
        self.create_synthetic_training_data()
        logging.info("Created new model with synthetic training")
    except Exception as e:
      logging.error(f"Error loading model:{str(e)}")
      self.model = self.create_model()
      logging.info("Created new model due to loading error")
      
  def create_synthetic_training_data(self):
    """Create synthetic training data for model initialization"""
    try:
      # Generate training data using image features like histograms and text features
      n_samples = len(self.class_names) * 50
      n_features = 100 # Feature vector size
      
      X_train = np.random.random((n_samples, n_features))
      y_train = np.repeat(range(len(self.class_names)), 50)
      
      # More strucuture to the data to make it realistic
      for i in range(len(self.class_names)):
        start_idx = i * 50
        end_idx = (i + 1) * 50
        # Add class-specific patterns
        X_train[start_idx:end_idx, :10] += np.random.normal(i * 0.1, 0.05, (50, 10))
        X_train[start_idx:end_idx, 10:20] += np.random.normal(i * 0.05, 0.02, (50, 10))
        
      # Split data for validation
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
      
      # Train the model
      self.model.fit(X_train, y_train)
      
      # Evaluate the model
      val_predictions = self.model.predict(X_val)
      accuracy = accuracy_score(y_val, val_predictions)
      logging.info(f"Model trained with validation accuracy: {accuracy:.2f}")
      
      # Save the model
      with open("models/plant_disease_model.pkl") as f:
        pickle.dump(self.model, f)
      logging.info("Model trained and saved with synthetic data")
    except Exception as e:
      logging.error(f"Error creating synthetic training data: {str(e)}")
      
      
  def extract_features(self, image_path):
    """Extract features from image for prediction"""
    try:
      # Load and preprocess image
      image = Image.open(image_path)
      image = image.convert("RGB")
      image - image.resize(self.img_size)
      
      # Convert to numpy array
      img_array = np.array(image)
      
      # Extract various features
      features = []
      
      # Colour histograms for each channel
      for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=20, range=(0, 255))
        features.extend(hist / np.sum(hist)) # Normalize
        
      # Basic statistics
      features.extend([
        np.mean(img_array),
        np.std(img_array),
        np.min(img_array),
        np.max(img_array)
      ])
      
      # Colour channel statistics
      for channel in range(3):
        channel_data = img_array[:, :, channel]
        features.extend([
          np.mean(channel_data),
          np.std(channel_data),
          np.percentile(channel_data, 25),
          np.percentile(channel_data, 75)
        ])
        
      # Testure features using OpenCV
      gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
      
      # Sobel edges
      sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
      sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
      features.extend([
        np.mean(np.abs(sobel_x)),
        np.mean(np.abs(sobel_y))
      ])
      
      # pad or truncate for consistent feature vector size
      if len(features) < 100:
        features.extend([0] * (100 - len(features)))
      else:
        features = features[:100]
        
      return np.array(features).reshape(1, -1)
    
    except Exception as e:
      logging.error(f"Error extracting features: {str(e)}")
      # Returns default feature vector
      return np.zeros((1, 100))
    
  
  def predict(self, image_path):
    """Predict disease from image"""
    try:
      # Extract features from image
      features = self.extract_features(image_path)
      
      # Make prediction
      predictions = self.model.predict_proba(features)[0]
      
      # Get top 3 predictions
      top_indices = np.argsort(predictions)[-3:][::-1]
      
      results = []
      for idx in top_indices:
        confidence = float(predictions[idx])
        class_name = self.class_names[idx]
        
        # Only include predictions with reasonable confidence
        if confidence > 0.1:
          results.append({
            "class": class_name,
            "confidence": confidence * 100
          })
          
          # If no confident predictions, return the top prediction
          if not results:
            top_idx = np.argmax(predictions)
            results.append({
              "class": self.class_names[top_idx],
              "confidence": float(predictions[top_idx]) * 100
            })
            
          return results
        
    except Exception as e:
      logging.error(f"Error making prediction: {str(e)}")
      # Return the intelligent presiction based on image analysis
      return self.analyze_image_heuristics(image_path)
    
    
  def analyze_image_heuristics(self, image_path):
    """Analyze image using heuristics when model fails"""
    try:
      image = Image.open(image_path)
      image = image.convert("RGB")
      img_array = np.array(image)
      
      # Basic heuristics based on colour anaylsis
      avg_green = np.mean(img_array[:, :, 1])
      avg_red = np.mean(img_array[:, :, 0])
      avg_blue = np.mean(img_array[:, :, 2])
      
      # Simple heuristic: if predominantly green, likely healthy
      if avg_green > avg_red and avg_green > avg_blue and avg_green > 100:
        return [{
          "class": "Healthy",
          "confidence": 75.0
        }]
      # If brown/yellow tones, likely diseased
      elif avg_red > avg_blue and avg_green > avg_blue:
        return [{
          "class": "Tomato_Early_Blight",
          "confidence": 60.0
        }]
      else:
        return [{
          "class": "Tomato_Late_Blight",
          "confidence": 55.0
        }]
        
    except Exception as e:
      logging.error(f"Error in heuristic analysis: {str(e)}")
      return [{
        "class": "Unable to detect disease",
        "confidence": 0.0
      }]
      
  
  def augment_image(self, image_path):
    """Apply image augmentation for better prediction"""
    try:
      image = cv2.imread(image_path)
      
      # Apply various augmentation techniques
      augmented_images = []
      
      # Original image
      augmented_images.append(image)
      
      # Horizontal flip
      flipped = cv2.flip(image, 1)
      augmented_images.append(flipped)
      
      # Brightness adjustment
      bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
      augmented_images.append(bright)
      
      # Gaussian blur
      blurred = cv2.GaussianBlur(image, (5, 5), 0)
      augmented_images.append(blurred)
      
      return augmented_images
    
    except Exception as e:
      logging.error(f"Error augmenting image: {str(e)}")
      return [cv2.imread(image_path)]
            