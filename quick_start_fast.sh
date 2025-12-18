#!/bin/bash

# Quick Start Script for Fast Training (Under 2 Hours)
# This script helps you get started with the fastest training option

echo "================================================================================"
echo "🚀 FAST LEAF DISEASE DETECTION TRAINING - Quick Start"
echo "================================================================================"
echo ""
echo "This script will help you train the model in UNDER 2 HOURS with 90-93% accuracy"
echo ""

# Check Python
echo "Checking Python installation..."
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Check GPU
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
    echo ""
    echo "⚡ GPU training will be MUCH faster than CPU (20-30x speedup)"
else
    echo "⚠️  No GPU detected. Training will be slower on CPU."
    echo "   Estimated time: 10-20 hours on CPU vs 1-2 hours on GPU"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check if dependencies are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  TensorFlow not found. Installing dependencies..."
    echo ""
    
    read -p "Install required packages? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing packages (this may take a few minutes)..."
        $PYTHON_CMD -m pip install -r requirements.txt
        
        if [ $? -eq 0 ]; then
            echo "✓ Dependencies installed successfully!"
        else
            echo "❌ Failed to install dependencies. Please install manually:"
            echo "   pip install -r requirements.txt"
            exit 1
        fi
    else
        echo "Please install dependencies manually and run this script again:"
        echo "  pip install -r requirements.txt"
        exit 0
    fi
else
    echo "✓ TensorFlow is installed"
fi
echo ""

# Check dataset
echo "Checking dataset..."
DATASET_PATH="/workspaces/Leaf_Disease_Detection/dataset"

if [ ! -d "$DATASET_PATH" ]; then
    echo "⚠️  Dataset not found at: $DATASET_PATH"
    echo ""
    echo "Please ensure your dataset is organized as:"
    echo "  dataset/"
    echo "    ├── train/"
    echo "    ├── val/"
    echo "    └── test/"
    echo ""
    read -p "Do you have the dataset in a different location? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter the full path to your dataset directory: " CUSTOM_PATH
        if [ -d "$CUSTOM_PATH" ]; then
            echo ""
            echo "⚠️  You'll need to update the BASE_DIR path in train_model_fast.py"
            echo "   Change line 30 to: BASE_DIR = '$CUSTOM_PATH'"
            echo ""
            read -p "Press Enter to continue..."
        else
            echo "❌ Directory not found: $CUSTOM_PATH"
            exit 1
        fi
    else
        echo "Please set up the dataset first."
        exit 0
    fi
else
    echo "✓ Dataset found at: $DATASET_PATH"
fi
echo ""

# Summary
echo "================================================================================"
echo "📋 TRAINING SUMMARY"
echo "================================================================================"
echo ""
echo "  Model:          MobileNetV3Small (optimized for speed)"
echo "  Expected Time:  60-100 minutes (1-1.7 hours)"
echo "  Expected Acc:   90-93% test accuracy"
echo "  Image Size:     224x224"
echo "  Batch Size:     64"
echo "  Max Epochs:     30 (early stopping enabled)"
echo ""
echo "  Features:"
echo "    ✓ Mixed precision training (GPU acceleration)"
echo "    ✓ Aggressive early stopping"
echo "    ✓ Optimized hyperparameters"
echo "    ✓ Streamlined augmentation"
echo ""
echo "================================================================================"
echo ""

# Ask to proceed
read -p "Ready to start training? (y/n): " -n 1 -r
echo
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled. To start later, run:"
    echo "  python train_model_fast.py"
    exit 0
fi

# Start training
echo "================================================================================"
echo "🚀 STARTING FAST TRAINING"
echo "================================================================================"
echo ""
echo "Training will begin now. This will take approximately 1-1.7 hours."
echo ""
echo "💡 Tips while training:"
echo "  • You can monitor progress in real-time"
echo "  • Press Ctrl+C to stop training (best model will be saved)"
echo "  • Training will auto-stop if accuracy plateaus"
echo "  • Watch for the 'target_met' indicator at the end"
echo ""
echo "================================================================================"
echo ""

# Run training
$PYTHON_CMD train_model_fast.py

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo "================================================================================"
    echo ""
    echo "Your trained models are saved in the 'models/' directory:"
    echo "  • models/best_model_fast_finetuned.h5  - Best model (use for predictions)"
    echo "  • models/final_model_fast.h5            - Final model"
    echo "  • models/class_indices_fast.json        - Class mappings"
    echo "  • models/training_summary_fast.json     - Training statistics"
    echo ""
    echo "Training plots are in 'plots/training_history_fast.png'"
    echo ""
    echo "================================================================================"
    echo "NEXT STEPS:"
    echo "================================================================================"
    echo ""
    echo "1. Evaluate your model:"
    echo "   python evaluate_model.py"
    echo ""
    echo "2. Make predictions:"
    echo "   python predict.py path/to/your/image.jpg"
    echo ""
    echo "3. View training curves:"
    echo "   Open plots/training_history_fast.png"
    echo ""
    echo "4. Check training statistics:"
    echo "   cat models/training_summary_fast.json"
    echo ""
    echo "================================================================================"
    echo "🎉 Congratulations! Your model is ready to use!"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "❌ TRAINING FAILED"
    echo "================================================================================"
    echo ""
    echo "Please check the error messages above for details."
    echo ""
    echo "Common issues:"
    echo "  • Out of memory: Reduce BATCH_SIZE in train_model_fast.py"
    echo "  • Dataset not found: Check dataset path"
    echo "  • GPU not working: Check CUDA installation with 'nvidia-smi'"
    echo ""
    echo "For help, see: FAST_TRAINING_GUIDE.md"
    echo "================================================================================"
    exit 1
fi
