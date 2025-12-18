"""
Prediction script for Plant Leaf Disease Detection
Load trained model and make predictions on new images
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

class LeafDiseasePredictor:
    def __init__(self, model_path='models/best_model_finetuned.h5', 
                 class_indices_path='models/class_indices.json',
                 img_size=300):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the trained model
            class_indices_path: Path to class indices JSON file
            img_size: Image size for preprocessing
        """
        self.img_size = img_size
        
        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping (index to class name)
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        print(f"✓ Loaded {len(self.class_indices)} disease classes")
    
    def preprocess_image(self, img_path):
        """
        Preprocess an image for prediction
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def predict(self, img_path, top_k=3):
        """
        Make prediction on a single image
        
        Args:
            img_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Preprocess image
        img_array = self.preprocess_image(img_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'predictions': [],
            'image_path': img_path
        }
        
        for idx in top_indices:
            class_name = self.idx_to_class[idx]
            confidence = float(predictions[idx])
            results['predictions'].append({
                'class': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            })
        
        return results
    
    def predict_and_visualize(self, img_path, top_k=3, save_path=None):
        """
        Make prediction and visualize the results
        
        Args:
            img_path: Path to the image file
            top_k: Number of top predictions to display
            save_path: Optional path to save the visualization
        """
        # Get predictions
        results = self.predict(img_path, top_k)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Display image
        img = Image.open(img_path)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        
        # Display predictions
        classes = [pred['class'] for pred in results['predictions']]
        confidences = [pred['confidence'] for pred in results['predictions']]
        
        # Shorten class names for display
        display_classes = [cls.replace('___', ' - ').replace('_', ' ') for cls in classes]
        
        colors = ['green' if confidences[0] > 0.9 else 'orange' if confidences[0] > 0.7 else 'red']
        colors.extend(['lightblue'] * (len(classes) - 1))
        
        bars = ax2.barh(display_classes, confidences, color=colors)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title(f'Top {top_k} Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        
        # Add percentage labels
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(conf + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{conf*100:.1f}%', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
        
        # Print results
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        for i, pred in enumerate(results['predictions'], 1):
            print(f"{i}. {pred['class']}")
            print(f"   Confidence: {pred['confidence_percent']}")
            print()
    
    def predict_batch(self, image_folder, output_file='predictions.json'):
        """
        Make predictions on multiple images in a folder
        
        Args:
            image_folder: Path to folder containing images
            output_file: Path to save results JSON
        """
        results = []
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.endswith(valid_extensions)]
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                prediction = self.predict(img_path, top_k=1)
                results.append({
                    'filename': img_file,
                    'predicted_class': prediction['predictions'][0]['class'],
                    'confidence': prediction['predictions'][0]['confidence_percent']
                })
                print(f"✓ {img_file}: {prediction['predictions'][0]['class']}")
            except Exception as e:
                print(f"✗ Error processing {img_file}: {e}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Batch predictions saved to {output_file}")
        return results


def main():
    """
    Example usage of the predictor
    """
    import sys
    
    # Initialize predictor
    predictor = LeafDiseasePredictor()
    
    if len(sys.argv) > 1:
        # Predict on provided image path
        img_path = sys.argv[1]
        if os.path.isfile(img_path):
            predictor.predict_and_visualize(img_path, top_k=3)
        elif os.path.isdir(img_path):
            predictor.predict_batch(img_path)
        else:
            print(f"Error: {img_path} is not a valid file or directory")
    else:
        # Example: predict on a test image
        test_dir = '/workspaces/Leaf_Disease_Detection/dataset/test'
        
        # Get first available test image
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                images = os.listdir(subdir_path)
                if images:
                    test_image = os.path.join(subdir_path, images[0])
                    print(f"Example prediction on: {test_image}")
                    predictor.predict_and_visualize(test_image, top_k=5, 
                                                    save_path='plots/example_prediction.png')
                    break


if __name__ == "__main__":
    main()
