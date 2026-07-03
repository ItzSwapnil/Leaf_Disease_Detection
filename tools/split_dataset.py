import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    random.seed(42)
    
    dataset_dir = Path("/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/dataset")
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    # Temporary output directories
    train_new = dataset_dir / "train_new"
    val_new = dataset_dir / "val_new"
    test_new = dataset_dir / "test_new"
    
    for d in [train_new, val_new, test_new]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all class folders
    all_classes = set()
    if train_dir.exists():
        all_classes.update(d.name for d in train_dir.iterdir() if d.is_dir())
    if val_dir.exists():
        all_classes.update(d.name for d in val_dir.iterdir() if d.is_dir())
        
    all_classes = sorted(list(all_classes))
    print(f"Found {len(all_classes)} unique classes.")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for cls in tqdm(all_classes, desc="Splitting classes"):
        cls_files = []
        
        # Collect from train
        cls_train_dir = train_dir / cls
        if cls_train_dir.exists():
            cls_files.extend(cls_train_dir.glob("*"))
            
        # Collect from val
        cls_val_dir = val_dir / cls
        if cls_val_dir.exists():
            cls_files.extend(cls_val_dir.glob("*"))
            
        # Filter files only
        cls_files = [f for f in cls_files if f.is_file()]
        random.shuffle(cls_files)
        
        n_total = len(cls_files)
        if n_total == 0:
            continue
            
        # Stratified counts: 80% train, 10% val, 10% test
        n_val = max(1, int(n_total * 0.10))
        n_test = max(1, int(n_total * 0.10))
        
        # Edge cases for very small classes
        if n_total < 3:
            n_val = 1 if n_total >= 2 else 0
            n_test = 1 if n_total >= 3 else 0
            
        n_train = n_total - n_val - n_test
        
        train_files = cls_files[:n_train]
        val_files = cls_files[n_train:n_train + n_val]
        test_files = cls_files[n_train + n_val:]
        
        # Create output class directories
        (train_new / cls).mkdir(exist_ok=True)
        (val_new / cls).mkdir(exist_ok=True)
        (test_new / cls).mkdir(exist_ok=True)
        
        # Copy files to preserve original data in case of failure
        for f in train_files:
            shutil.copy(f, train_new / cls / f.name)
        for f in val_files:
            shutil.copy(f, val_new / cls / f.name)
        for f in test_files:
            shutil.copy(f, test_new / cls / f.name)
            
        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)
        
    print(f"\nDataset Split Statistics:")
    print(f"  - Train: {total_train} files")
    print(f"  - Validation: {total_val} files")
    print(f"  - Test: {total_test} files")
    print(f"  - Total: {total_train + total_val + total_test} files")
    
    # Verify everything looks good before overwriting
    if total_train > 0 and total_val > 0 and total_test > 0:
        print("\nReplacing old directories...")
        # Remove old train and val
        if train_dir.exists():
            shutil.rmtree(train_dir)
        if val_dir.exists():
            shutil.rmtree(val_dir)
            
        # Rename new directories
        train_new.rename(train_dir)
        val_new.rename(val_dir)
        test_new.rename(dataset_dir / "test")
        print("Success! Dataset is now properly partitioned into train, val, and test splits.")
    else:
        print("Error: One of the splits is empty. Aborting directory swap.")
        # Cleanup new directories
        shutil.rmtree(train_new)
        shutil.rmtree(val_new)
        shutil.rmtree(test_new)

if __name__ == "__main__":
    main()
