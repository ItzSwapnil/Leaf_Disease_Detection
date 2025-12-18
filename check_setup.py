"""
Simple script to verify the system setup and run a quick test
"""

import os
import sys

def check_imports():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_gpu():
    """Check GPU availability"""
    import tensorflow as tf
    
    print("\nChecking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Test GPU
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                c = tf.matmul(a, b)
            print("✅ GPU test successful!")
        except:
            print("⚠️  GPU detected but test failed")
    else:
        print("⚠️  No GPU detected - training will be slow")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")


def check_dataset():
    """Check dataset structure"""
    print("\nChecking dataset...")
    
    base_dir = '/workspaces/Leaf_Disease_Detection/dataset'
    required_dirs = ['train', 'val', 'test']
    
    if not os.path.exists(base_dir):
        print(f"❌ Dataset directory not found: {base_dir}")
        return False
    
    all_good = True
    for split in required_dirs:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            # Count classes
            classes = [d for d in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, d))]
            # Count images
            num_images = sum([len(os.listdir(os.path.join(split_dir, cls))) 
                             for cls in classes])
            print(f"✓ {split:5s}: {len(classes)} classes, {num_images:,} images")
        else:
            print(f"✗ {split:5s}: Directory not found")
            all_good = False
    
    if all_good:
        print("\n✅ Dataset structure looks good!")
    else:
        print("\n❌ Dataset incomplete")
    
    return all_good


def check_disk_space():
    """Check available disk space"""
    import shutil
    
    print("\nChecking disk space...")
    
    total, used, free = shutil.disk_usage("/")
    
    gb = 1024 ** 3
    print(f"Total: {total / gb:.1f} GB")
    print(f"Used:  {used / gb:.1f} GB")
    print(f"Free:  {free / gb:.1f} GB")
    
    if free < 10 * gb:
        print("⚠️  Low disk space - at least 10GB recommended for training")
    else:
        print("✅ Sufficient disk space")


def create_sample_config():
    """Create a sample configuration file"""
    config = """
# Training Configuration
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 46

# Paths
TRAIN_DIR = '/workspaces/Leaf_Disease_Detection/dataset/train'
VAL_DIR = '/workspaces/Leaf_Disease_Detection/dataset/val'
TEST_DIR = '/workspaces/Leaf_Disease_Detection/dataset/test'

# Model
BASE_MODEL = 'EfficientNetB3'  # Options: EfficientNetB0, EfficientNetB3, MobileNetV2

# Callbacks
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
"""
    
    with open('config.py', 'w') as f:
        f.write(config)
    
    print("\n✅ Sample configuration file created: config.py")


def main():
    print("="*80)
    print("Plant Leaf Disease Detection - System Check")
    print("="*80)
    
    # Run all checks
    checks_passed = 0
    total_checks = 4
    
    if check_imports():
        checks_passed += 1
    
    try:
        check_gpu()
        checks_passed += 1
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")
    
    if check_dataset():
        checks_passed += 1
    
    try:
        check_disk_space()
        checks_passed += 1
    except Exception as e:
        print(f"⚠️  Disk space check failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print(f"System Check Summary: {checks_passed}/{total_checks} checks passed")
    print("="*80)
    
    if checks_passed == total_checks:
        print("\n🎉 System is ready for training!")
        print("\nNext steps:")
        print("  1. Review configuration in train_model.py")
        print("  2. Run: python train_model.py")
        print("  3. Monitor with: tensorboard --logdir logs")
        print("\nOr use the quick start script: ./quick_start.sh")
    else:
        print("\n⚠️  Some issues detected. Please fix them before training.")
    
    # Create sample config
    if not os.path.exists('config.py'):
        create_sample_config()


if __name__ == "__main__":
    main()
