#!/usr/bin/env python3
"""
Extract embeddings from trained jewelry model for inference
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json

class L2Norm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=self.eps)

class JewelryDINOv2(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        # Load pretrained DINOv2 ViT-L/14
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        # FREEZE the backbone - this is crucial!
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Only train a lightweight projection head
        backbone_dim = 1024  # ViT-L/14 output dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),  # Add batch norm for stability
            nn.Dropout(0.15),
            nn.Linear(512, emb_dim),
            L2Norm()  # L2 normalize embeddings
        )
        
    def forward(self, x):
        with torch.no_grad():  # No gradients through backbone
            features = self.backbone(x)
            
        # Handle different output formats
        if isinstance(features, dict):
            features = features.get('x_norm_clstoken', features.get('cls_token', features))
        if features.dim() == 3:
            features = features[:, 0, :]  # Take CLS token
            
        embeddings = self.projection(features)
        return embeddings

def get_inference_transform():
    """Inference transform - no augmentation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def extract_embeddings(model_path, data_path, output_dir):
    """Extract all embeddings from the trained model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = JewelryDINOv2(emb_dim=256).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best Recall@1: {checkpoint['recall_metrics'][1]:.4f}")
    
    # Create dataset
    transform = get_inference_transform()
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")
    
    # Extract embeddings
    all_embeddings = []
    all_labels = []
    all_filenames = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Processing batches")):
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Get labels and filenames for this batch
            batch_start = batch_idx * dataloader.batch_size
            batch_labels = labels.numpy()
            all_labels.extend(batch_labels)
            
            for i, label in enumerate(batch_labels):
                sample_idx = batch_start + i
                if sample_idx < len(dataset.samples):
                    filename = dataset.samples[sample_idx][0]
                    all_filenames.append(os.path.relpath(filename, data_path))
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    print(f"\nGenerated embeddings shape: {all_embeddings.shape}")
    print(f"Total samples: {len(all_embeddings)}")
    
    # Save embeddings and metadata
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving files...")
    np.save(os.path.join(output_dir, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
    
    with open(os.path.join(output_dir, 'filenames.txt'), 'w') as f:
        f.write('\n'.join(all_filenames))
    
    # Save class mappings
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    with open(os.path.join(output_dir, 'class_mappings.json'), 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'embedding_dim': all_embeddings.shape[1],
            'total_samples': len(all_embeddings),
            'model_performance': checkpoint['recall_metrics']
        }, f, indent=2)
    
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - embeddings.npy: {all_embeddings.shape} embeddings")
    print(f"  - labels.npy: {len(all_labels)} labels")
    print(f"  - filenames.txt: {len(all_filenames)} filenames")
    print(f"  - class_mappings.json: metadata")
    
    # Print class distribution
    print(f"\nClass distribution:")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        class_name = idx_to_class[label]
        print(f"  {class_name}: {count} images")
    
    print("\nEmbedding extraction complete! Ready for inference.")

if __name__ == '__main__':
    model_path = 'jewelry_model/best_model.pth'
    data_path = 'Pictures/3D'
    output_dir = 'jewelry_model'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        exit(1)
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found!")
        exit(1)
    
    extract_embeddings(model_path, data_path, output_dir)