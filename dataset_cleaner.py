#!/usr/bin/env python3
"""
Script to clean and validate jewelry dataset before training.
Identifies and removes corrupted/invalid image files.
"""

import os
import shutil
from PIL import Image
from tqdm import tqdm
import argparse

def is_valid_image(filepath):
    """Check if an image file is valid and can be opened"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify the image
        
        # Try to actually load it (verify doesn't catch all issues)
        with Image.open(filepath) as img:
            img.load()
        
        return True
    except Exception as e:
        return False

def clean_dataset(data_path, backup_corrupted=True):
    """Clean dataset by removing or backing up corrupted images"""
    
    # Create backup directory for corrupted files
    if backup_corrupted:
        backup_dir = os.path.join(os.path.dirname(data_path), "corrupted_images_backup")
        os.makedirs(backup_dir, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    total_files = 0
    corrupted_files = []
    
    print(f"Scanning dataset: {data_path}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in valid_extensions:
                total_files += 1
    
    print(f"Found {total_files} image files. Validating...")
    
    # Validate all images
    with tqdm(total=total_files, desc="Validating images") as pbar:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                filepath = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in valid_extensions:
                    if not is_valid_image(filepath):
                        corrupted_files.append(filepath)
                        pbar.set_description(f"Found corrupted: {os.path.basename(filepath)}")
                    pbar.update(1)
    
    print(f"\nValidation complete!")
    print(f"Total files: {total_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\nCorrupted files found:")
        for f in corrupted_files:
            print(f"  - {f}")
        
        if backup_corrupted:
            print(f"\nBacking up corrupted files to: {backup_dir}")
            for filepath in corrupted_files:
                # Recreate directory structure in backup
                rel_path = os.path.relpath(filepath, data_path)
                backup_filepath = os.path.join(backup_dir, rel_path)
                os.makedirs(os.path.dirname(backup_filepath), exist_ok=True)
                
                # Move corrupted file to backup
                shutil.move(filepath, backup_filepath)
                print(f"  Moved: {os.path.basename(filepath)}")
        else:
            print("\nRemoving corrupted files...")
            for filepath in corrupted_files:
                os.remove(filepath)
                print(f"  Removed: {os.path.basename(filepath)}")
    
    # Check for empty directories and remove them
    empty_dirs = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        if not dirs and not files and root != data_path:
            empty_dirs.append(root)
    
    if empty_dirs:
        print(f"\nRemoving {len(empty_dirs)} empty directories...")
        for empty_dir in empty_dirs:
            os.rmdir(empty_dir)
            print(f"  Removed empty dir: {empty_dir}")
    
    print(f"\nDataset cleaning complete!")
    print(f"Clean images remaining: {total_files - len(corrupted_files)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for class_dir in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.splitext(f)[1].lower() in valid_extensions]
            print(f"  {class_dir}: {len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description="Clean jewelry dataset")
    parser.add_argument('--data-path', default='Pictures/3D', help='Path to dataset')
    parser.add_argument('--no-backup', action='store_true', help='Delete corrupted files instead of backing up')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        return
    
    clean_dataset(args.data_path, backup_corrupted=not args.no_backup)

if __name__ == '__main__':
    main()