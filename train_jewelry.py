#!/usr/bin/env python3
"""
Improved DINOv2 training for jewelry images.
Key improvements:
1. Frozen backbone + trainable head only
2. Triplet loss instead of SupCon (works better with small datasets)  
3. Single-stage training with ViT-L/14 directly
4. Conservative data augmentation for jewelry
5. Optimized for RTX 5070 Ti (16GB VRAM)
6. Enhanced for cross-category similarity matching
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import logging
import json
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripletLoss(nn.Module):
    """More stable than SupCon for small datasets with numerical stability improvements"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # Create positive and negative pairs
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        # For each anchor, find hardest positive and hardest negative
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = ~mask_pos
        
        # Remove self-comparison
        mask_pos.fill_diagonal_(False)
        
        losses = []
        for i in range(len(embeddings)):
            if not mask_pos[i].any() or not mask_neg[i].any():
                continue
                
            # Hardest positive (most distant same class)
            pos_dists = pairwise_dist[i][mask_pos[i]]
            hardest_pos = pos_dists.max()
            
            # Hardest negative (closest different class)  
            neg_dists = pairwise_dist[i][mask_neg[i]]
            hardest_neg = neg_dists.min()
            
            # Add small epsilon for numerical stability
            loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=1e-6)
            losses.append(loss)
            
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=embeddings.device)

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

class L2Norm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=self.eps)

def get_jewelry_transforms():
    """Conservative augmentations for jewelry - avoid destroying important details"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.8, 1.2)),  # Slightly less aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),
        # Very mild color jittering - jewelry color/materials are important
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomRotation(15),  # Small rotations only
        # Add some blur occasionally to help with different photo qualities
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                loss = criterion(embeddings, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
    return total_loss / num_batches

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, top_k=[1, 5, 10]):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device, non_blocking=True)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
        
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Compute recall@k
    similarities = np.dot(all_embeddings, all_embeddings.T)
    np.fill_diagonal(similarities, -1)  # Remove self-similarity
    
    recall_at_k = {k: 0 for k in top_k}
    
    for i in range(len(all_embeddings)):
        # Get top-k most similar
        top_k_idx = np.argsort(similarities[i])[::-1]
        query_label = all_labels[i]
        
        # Check recall@k for different k values
        for k in top_k:
            if k <= len(top_k_idx) and query_label in all_labels[top_k_idx[:k]]:
                recall_at_k[k] += 1
                
    # Normalize by total number of queries
    for k in top_k:
        recall_at_k[k] /= len(all_embeddings)
    
    return recall_at_k

def create_stratified_split(dataset, val_ratio=0.2, random_seed=42):
    """Create stratified train/val split to ensure balanced classes"""
    import random
    from collections import defaultdict
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Group samples by class
    class_samples = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_samples[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for label, samples in class_samples.items():
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))  # At least 1 sample per class in val
        val_indices.extend(samples[:n_val])
        train_indices.extend(samples[n_val:])
    
    return train_indices, val_indices

def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 for jewelry similarity search")
    parser.add_argument('--data-path', default='Pictures/3D', help='Path to jewelry dataset')
    parser.add_argument('--output-dir', default='./jewelry_dinov2_model')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (optimized for RTX 5070 Ti)')
    parser.add_argument('--epochs', type=int, default=35)   
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--emb-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-workers', type=int, default=12, help='Data loader workers')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--margin', type=float, default=0.3, help='Triplet loss margin')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
    
    args = parser.parse_args()
    
    # Verify the path structure
    if not os.path.exists(args.data_path):
        logger.error(f"Data path {args.data_path} does not exist!")
        return
    
    # Check if subdirectories exist
    subdirs = [d for d in os.listdir(args.data_path) 
               if os.path.isdir(os.path.join(args.data_path, d))]
    logger.info(f"Found {len(subdirs)} jewelry categories: {subdirs}")
    
    if len(subdirs) == 0:
        logger.error("No subdirectories found in data path!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data
    train_transform, val_transform = get_jewelry_transforms()
    
    # Create datasets
    full_dataset = ImageFolder(args.data_path, transform=train_transform)
    val_dataset = ImageFolder(args.data_path, transform=val_transform)
    
    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(f"Number of classes: {len(full_dataset.classes)}")
    
    # Create stratified split
    train_indices, val_indices = create_stratified_split(full_dataset, args.val_ratio)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset_subset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset_subset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model
    model = JewelryDINOv2(emb_dim=args.emb_dim).to(device)
    
    # Only projection parameters are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Loss and optimizer
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # Scheduler with warmup
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_recall_1 = 0
    training_history = []
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Evaluate  
        recall_metrics = evaluate_retrieval(model, val_loader, device, top_k=[1, 5, 10])
        
        # Log results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        for k, recall in recall_metrics.items():
            logger.info(f"Recall@{k}: {recall:.4f}")
        
        # Save training history
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            **{f'recall_{k}': v for k, v in recall_metrics.items()},
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_metrics)
        
        # Save best model
        if recall_metrics[1] > best_recall_1:
            best_recall_1 = recall_metrics[1]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'recall_metrics': recall_metrics,
                'epoch': epoch + 1,
                'args': args,
                'class_to_idx': full_dataset.class_to_idx
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"New best model saved! Recall@1: {recall_metrics[1]:.4f}")
        
        # Save latest model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'recall_metrics': recall_metrics,
            'epoch': epoch + 1,
            'args': args,
            'class_to_idx': full_dataset.class_to_idx
        }, os.path.join(args.output_dir, 'latest_model.pth'))
        
        scheduler.step()
        
    logger.info(f"\nTraining complete. Best Recall@1: {best_recall_1:.4f}")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save embeddings and metadata for inference
    logger.info("Saving embeddings for inference...")
    save_embeddings_for_inference(model, full_dataset, val_transform, device, args.output_dir)

def save_embeddings_for_inference(model, dataset, transform, device, output_dir):
    """Save all embeddings and metadata for similarity search"""
    from torch.utils.data import DataLoader
    
    # Create dataset with transform for inference
    inference_dataset = ImageFolder(dataset.root, transform=transform)
    inference_loader = DataLoader(inference_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    
    model.eval()
    all_embeddings = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(inference_loader, desc="Computing final embeddings")):
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Get labels and filenames for this batch
            batch_start = batch_idx * inference_loader.batch_size
            batch_labels = labels.numpy()
            all_labels.extend(batch_labels)
            
            for i, label in enumerate(batch_labels):
                sample_idx = batch_start + i
                if sample_idx < len(inference_dataset.samples):
                    filename = inference_dataset.samples[sample_idx][0]
                    all_filenames.append(os.path.basename(filename))
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
    
    # Save embeddings and metadata
    np.save(os.path.join(output_dir, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
    
    with open(os.path.join(output_dir, 'filenames.txt'), 'w') as f:
        f.write('\n'.join(all_filenames))
    
    # Save class mappings
    class_to_idx = inference_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    with open(os.path.join(output_dir, 'class_mappings.json'), 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'embedding_dim': all_embeddings.shape[1],
            'total_samples': len(all_embeddings)
        }, f, indent=2)
    
    logger.info(f"Saved {len(all_embeddings)} embeddings for inference")
    logger.info(f"Classes found: {list(class_to_idx.keys())}")

if __name__ == '__main__':
    main()