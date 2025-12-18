#!/usr/bin/env python3
"""
Test script to validate train_model_fast.py configuration
Tests the logic and configuration without requiring full TensorFlow installation
"""

import sys
import os

def test_fast_training_config():
    """Test the configuration values in train_model_fast.py"""
    
    print("="*60)
    print("Testing Fast Training Configuration")
    print("="*60)
    
    # Expected configuration values
    expected_config = {
        'IMG_SIZE': 160,
        'BATCH_SIZE': 64,
        'EPOCHS_PHASE1': 8,
        'EPOCHS_PHASE2': 7,
        'TOTAL_EPOCHS': 15,
        'NUM_CLASSES': 46,
        'LEARNING_RATE': 0.001
    }
    
    # Read the file
    script_path = 'train_model_fast.py'
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script_path} not found!")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Validate key configurations
    all_passed = True
    
    for key, expected_value in expected_config.items():
        if key == 'TOTAL_EPOCHS':
            # TOTAL_EPOCHS is computed from EPOCHS_PHASE1 + EPOCHS_PHASE2
            if "TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2" in content:
                print(f"✓ {key}: {expected_value} (computed)")
            else:
                print(f"❌ {key}: Expected computation not found")
                all_passed = False
        elif f"{key} = {expected_value}" in content:
            print(f"✓ {key}: {expected_value}")
        else:
            print(f"❌ {key}: Expected {expected_value} but not found in correct format")
            all_passed = False
    
    # Check for key features
    features = {
        'Mixed Precision': "set_global_policy('mixed_float16')",
        'MobileNetV2': "from tensorflow.keras.applications import MobileNetV2",
        'GlobalAveragePooling2D': "GlobalAveragePooling2D()",
        'ModelCheckpoint': "ModelCheckpoint(",
        'EarlyStopping': "EarlyStopping(",
        'Two-Phase Training': "history_phase1 = model.fit(",
        'Fine-tuning': "base_model.trainable = True",
        'Save Statistics': "training_stats_fast.json",
        'Save Model': "best_model_fast_finetuned.h5",
        'Plot Generation': "plt.savefig('plots/training_history_fast.png'"
    }
    
    print("\nFeature Checks:")
    print("-" * 60)
    
    for feature_name, feature_code in features.items():
        if feature_code in content:
            print(f"✓ {feature_name}")
        else:
            print(f"❌ {feature_name}: Not found")
            all_passed = False
    
    # Validate time tracking
    time_tracking = [
        'training_start_time = time.time()',
        'phase1_start = time.time()',
        'phase2_start = time.time()',
        'total_training_time = time.time() - training_start_time'
    ]
    
    print("\nTime Tracking:")
    print("-" * 60)
    
    for check in time_tracking:
        if check in content:
            print(f"✓ {check}")
        else:
            print(f"❌ {check}: Not found")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nConfiguration Summary:")
        print(f"  • Image Size: 160x160 (optimized for speed)")
        print(f"  • Batch Size: 64 (increased throughput)")
        print(f"  • Total Epochs: 15 (8 + 7)")
        print(f"  • Mixed Precision: Enabled")
        print(f"  • Model: MobileNetV2")
        print(f"  • Expected Time: 60-90 minutes")
        print(f"  • Expected Accuracy: 92-95%")
        print("="*60)
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return False

def test_documentation():
    """Test that documentation files exist"""
    
    print("\n" + "="*60)
    print("Testing Documentation")
    print("="*60)
    
    docs = [
        'README_FAST_TRAINING.md',
        'README.md',
        'QUICK_REFERENCE.md'
    ]
    
    all_exist = True
    
    for doc in docs:
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            print(f"✓ {doc} ({size:,} bytes)")
        else:
            print(f"❌ {doc} not found")
            all_exist = False
    
    return all_exist

def test_file_structure():
    """Test that the script creates necessary directories"""
    
    print("\n" + "="*60)
    print("Testing File Structure Requirements")
    print("="*60)
    
    with open('train_model_fast.py', 'r') as f:
        content = f.read()
    
    required_dirs = ['models', 'logs', 'plots']
    
    for directory in required_dirs:
        if f"os.makedirs('{directory}', exist_ok=True)" in content:
            print(f"✓ Creates directory: {directory}")
        else:
            print(f"❌ Missing directory creation: {directory}")
            return False
    
    return True

def main():
    """Run all tests"""
    
    print("\n" + "="*60)
    print("FAST TRAINING SCRIPT VALIDATION")
    print("="*60)
    print("\nValidating train_model_fast.py and documentation...")
    print()
    
    results = []
    
    # Run tests
    results.append(("Configuration", test_fast_training_config()))
    results.append(("Documentation", test_documentation()))
    results.append(("File Structure", test_file_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
        print("\n📋 Next Steps:")
        print("  1. Ensure dataset is in /workspaces/Leaf_Disease_Detection/dataset/")
        print("  2. Run: python train_model_fast.py")
        print("  3. Training should complete in 60-90 minutes")
        print("  4. Expected accuracy: 92-95%")
        print("\n" + "="*60)
        return 0
    else:
        print("\n❌ Some validation tests failed.")
        print("Please review the output above for details.")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
