import argparse
import os
import logging
from ml_model import PlantDiseaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Evaluate Plant Disease Detection Model')
    parser.add_argument('--test_path', type=str, help='Path to test dataset', default="dataset/test")
    parser.add_argument('--model_path', type=str, default='models/plant_disease_cnn_model.keras', help='Path to trained model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_path):
        logging.error(f"Test dataset path {args.test_path} does not exist!")
        return
    
    if not os.path.exists(args.model_path):
        logging.error(f"Model path {args.model_path} does not exist!")
        return
    
    # Load model
    logging.info("Loading model...")
    model = PlantDiseaseModel()
    
    # Evaluate model
    logging.info(f"Evaluating model on test dataset: {args.test_path}")
    accuracy, report = model.evaluate_model(args.test_path)
    
    if accuracy:
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{report}")
    else:
        logging.error("Evaluation failed!")

if __name__ == "__main__":
    main()
