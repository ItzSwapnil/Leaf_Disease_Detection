"""
Comprehensive Model Evaluation Script
Generates detailed metrics, confusion matrix, and classification report
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

class ModelEvaluator:
    def __init__(self, model_path='models/best_model_finetuned.h5',
                 class_indices_path='models/class_indices.json',
                 test_dir='/workspaces/Leaf_Disease_Detection/dataset/test',
                 img_size=300,
                 batch_size=32):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to trained model
            class_indices_path: Path to class indices JSON
            test_dir: Path to test dataset
            img_size: Image size for preprocessing
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Load model
        print("Loading model...")
        self.model = load_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        self.class_names = [self.idx_to_class[i] for i in range(len(self.class_indices))]
        
        # Setup test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✓ Test samples: {self.test_generator.samples}")
        print(f"✓ Classes: {len(self.class_names)}")
    
    def evaluate_model(self):
        """
        Evaluate model on test set
        """
        print("\nEvaluating model on test set...")
        results = self.model.evaluate(self.test_generator, verbose=1)
        
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]*100:.2f}%")
        if len(results) > 2:
            print(f"Test Top-3 Accuracy: {results[2]*100:.2f}%")
        print("="*80)
        
        return results
    
    def generate_predictions(self):
        """
        Generate predictions for confusion matrix and classification report
        """
        print("\nGenerating predictions...")
        
        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        return y_true, y_pred, predictions
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='plots/confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        print("\nGenerating confusion matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot confusion matrix (simplified view - percentage)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Due to many classes, create a simplified heatmap
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                   ax=ax1, cbar_kws={'label': 'Percentage'})
        ax1.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Plot per-class accuracy
        sorted_indices = np.argsort(class_accuracy)
        sorted_classes = [self.class_names[i].replace('___', '\n').replace('_', ' ') 
                         for i in sorted_indices]
        sorted_accuracy = class_accuracy[sorted_indices]
        
        colors = ['red' if acc < 0.7 else 'orange' if acc < 0.85 else 'green' 
                 for acc in sorted_accuracy]
        
        ax2.barh(range(len(sorted_classes)), sorted_accuracy, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_classes)))
        ax2.set_yticklabels(sorted_classes, fontsize=7)
        ax2.set_xlabel('Accuracy', fontsize=12)
        ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax2.axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='85% threshold')
        ax2.axvline(x=0.70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        
        return cm, class_accuracy
    
    def generate_classification_report(self, y_true, y_pred, save_path='evaluation_report.txt'):
        """
        Generate detailed classification report
        """
        print("\nGenerating classification report...")
        
        # Generate report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(report)
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PLANT LEAF DISEASE DETECTION - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Set: {self.test_dir}\n")
            f.write(f"Total Test Samples: {len(y_true)}\n")
            f.write(f"Number of Classes: {len(self.class_names)}\n\n")
            f.write("="*80 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(report)
        
        print(f"✓ Classification report saved to {save_path}")
        
        return report
    
    def plot_top_errors(self, y_true, y_pred, top_n=10, save_path='plots/top_errors.png'):
        """
        Plot classes with highest error rates
        """
        print("\nAnalyzing error patterns...")
        
        # Calculate error rate per class
        cm = confusion_matrix(y_true, y_pred)
        correct = cm.diagonal()
        total = cm.sum(axis=1)
        error_rate = 1 - (correct / total)
        
        # Get top N classes with highest error rate
        top_error_indices = np.argsort(error_rate)[-top_n:][::-1]
        
        plt.figure(figsize=(12, 6))
        
        classes = [self.class_names[i].replace('___', '\n').replace('_', ' ') 
                  for i in top_error_indices]
        errors = error_rate[top_error_indices]
        
        colors = ['darkred' if e > 0.3 else 'red' if e > 0.2 else 'orange' for e in errors]
        
        bars = plt.barh(classes, errors, color=colors, alpha=0.7)
        plt.xlabel('Error Rate', fontsize=12)
        plt.title(f'Top {top_n} Classes with Highest Error Rate', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for bar, error in zip(bars, errors):
            plt.text(error + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{error*100:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error analysis saved to {save_path}")
    
    def generate_performance_summary(self, y_true, y_pred, class_accuracy, 
                                    save_path='plots/performance_summary.png'):
        """
        Generate comprehensive performance summary
        """
        print("\nGenerating performance summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall Accuracy Distribution
        ax1 = axes[0, 0]
        accuracy_ranges = ['<70%', '70-85%', '85-95%', '>95%']
        counts = [
            np.sum(class_accuracy < 0.7),
            np.sum((class_accuracy >= 0.7) & (class_accuracy < 0.85)),
            np.sum((class_accuracy >= 0.85) & (class_accuracy < 0.95)),
            np.sum(class_accuracy >= 0.95)
        ]
        colors_dist = ['red', 'orange', 'lightgreen', 'green']
        ax1.bar(accuracy_ranges, counts, color=colors_dist, alpha=0.7)
        ax1.set_ylabel('Number of Classes', fontsize=12)
        ax1.set_title('Class Accuracy Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, count in enumerate(counts):
            ax1.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
        
        # 2. Prediction Confidence Distribution
        ax2 = axes[0, 1]
        predictions = self.model.predict(self.test_generator, verbose=0)
        max_confidences = np.max(predictions, axis=1)
        ax2.hist(max_confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(max_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_confidences):.3f}')
        ax2.set_xlabel('Prediction Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Performance Metrics
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        accuracy = np.mean(y_true == y_pred)
        mean_class_acc = np.mean(class_accuracy)
        median_class_acc = np.median(class_accuracy)
        min_class_acc = np.min(class_accuracy)
        max_class_acc = np.max(class_accuracy)
        
        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*50}
        
        Overall Accuracy:           {accuracy*100:.2f}%
        Mean Per-Class Accuracy:    {mean_class_acc*100:.2f}%
        Median Per-Class Accuracy:  {median_class_acc*100:.2f}%
        Min Per-Class Accuracy:     {min_class_acc*100:.2f}%
        Max Per-Class Accuracy:     {max_class_acc*100:.2f}%
        
        Classes with >95% accuracy:  {np.sum(class_accuracy > 0.95)}
        Classes with >85% accuracy:  {np.sum(class_accuracy > 0.85)}
        Classes with <70% accuracy:  {np.sum(class_accuracy < 0.7)}
        
        Total Test Samples:         {len(y_true)}
        Correctly Classified:       {np.sum(y_true == y_pred)}
        Misclassified:             {np.sum(y_true != y_pred)}
        """
        
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
                verticalalignment='center')
        
        # 4. Best and Worst Performing Classes
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        best_indices = np.argsort(class_accuracy)[-5:][::-1]
        worst_indices = np.argsort(class_accuracy)[:5]
        
        performance_text = "BEST PERFORMING CLASSES\n" + "="*50 + "\n"
        for idx in best_indices:
            class_name = self.class_names[idx].replace('___', ' - ').replace('_', ' ')
            performance_text += f"{class_name[:40]:40s} {class_accuracy[idx]*100:5.1f}%\n"
        
        performance_text += "\n\nWORST PERFORMING CLASSES\n" + "="*50 + "\n"
        for idx in worst_indices:
            class_name = self.class_names[idx].replace('___', ' - ').replace('_', ' ')
            performance_text += f"{class_name[:40]:40s} {class_accuracy[idx]*100:5.1f}%\n"
        
        ax4.text(0.05, 0.5, performance_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center')
        
        plt.suptitle('Model Performance Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance summary saved to {save_path}")
    
    def run_complete_evaluation(self):
        """
        Run complete evaluation pipeline
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # Create output directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Basic evaluation
        self.evaluate_model()
        
        # 2. Generate predictions
        y_true, y_pred, predictions = self.generate_predictions()
        
        # 3. Confusion matrix
        cm, class_accuracy = self.plot_confusion_matrix(y_true, y_pred)
        
        # 4. Classification report
        self.generate_classification_report(y_true, y_pred)
        
        # 5. Error analysis
        self.plot_top_errors(y_true, y_pred)
        
        # 6. Performance summary
        self.generate_performance_summary(y_true, y_pred, class_accuracy)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  - plots/confusion_matrix.png")
        print("  - plots/top_errors.png")
        print("  - plots/performance_summary.png")
        print("  - evaluation_report.txt")
        print("="*80)


def main():
    """
    Run evaluation
    """
    evaluator = ModelEvaluator()
    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()
