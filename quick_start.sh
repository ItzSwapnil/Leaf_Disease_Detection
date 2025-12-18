#!/bin/bash

# Quick Start Script for Plant Leaf Disease Detection
# This script helps you set up and run the project

echo "======================================================================"
echo "   Plant Leaf Disease Detection System - Quick Start"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python $python_version detected"
else
    print_error "Python 3.8+ required. Found: Python $python_version"
    exit 1
fi

# Check if GPU is available
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "NVIDIA driver found but GPU not accessible"
    fi
else
    print_warning "No GPU detected. Training will be slow on CPU."
fi

# Install dependencies
echo ""
echo "======================================================================"
echo "Installing dependencies..."
echo "======================================================================"

if pip3 install -r requirements.txt; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Check dataset
echo ""
echo "======================================================================"
echo "Checking dataset..."
echo "======================================================================"

if [ -d "dataset/train" ] && [ -d "dataset/val" ] && [ -d "dataset/test" ]; then
    train_count=$(find dataset/train -type f | wc -l)
    val_count=$(find dataset/val -type f | wc -l)
    test_count=$(find dataset/test -type f | wc -l)
    
    print_status "Dataset found:"
    echo "  - Training images: $train_count"
    echo "  - Validation images: $val_count"
    echo "  - Test images: $test_count"
else
    print_error "Dataset folders not found!"
    echo "Please ensure dataset/train, dataset/val, and dataset/test folders exist."
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating output directories..."
mkdir -p models plots logs
print_status "Output directories created"

# Ask user what to do
echo ""
echo "======================================================================"
echo "What would you like to do?"
echo "======================================================================"
echo "1) Train the model (recommended first step)"
echo "2) Evaluate an existing model"
echo "3) Make predictions on test images"
echo "4) View TensorBoard (requires existing logs)"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "======================================================================"
        echo "Starting Model Training"
        echo "======================================================================"
        echo ""
        print_warning "This will take 2-4 hours with GPU, much longer with CPU"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python3 train_model.py
        else
            echo "Training cancelled."
        fi
        ;;
    2)
        echo ""
        echo "======================================================================"
        echo "Evaluating Model"
        echo "======================================================================"
        echo ""
        if [ -f "models/best_model_finetuned.h5" ]; then
            python3 evaluate_model.py
        else
            print_error "No trained model found. Please train the model first (option 1)."
        fi
        ;;
    3)
        echo ""
        echo "======================================================================"
        echo "Making Predictions"
        echo "======================================================================"
        echo ""
        if [ -f "models/best_model_finetuned.h5" ]; then
            python3 predict.py
        else
            print_error "No trained model found. Please train the model first (option 1)."
        fi
        ;;
    4)
        echo ""
        echo "======================================================================"
        echo "Starting TensorBoard"
        echo "======================================================================"
        echo ""
        if [ -d "logs" ] && [ "$(ls -A logs)" ]; then
            echo "TensorBoard will be available at: http://localhost:6006"
            tensorboard --logdir logs
        else
            print_error "No training logs found. Please train the model first (option 1)."
        fi
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "Done!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  - Train: python3 train_model.py"
echo "  - Evaluate: python3 evaluate_model.py"
echo "  - Predict: python3 predict.py <image_path>"
echo "  - TensorBoard: tensorboard --logdir logs"
echo ""
print_status "For more information, see README_MODEL.md"
