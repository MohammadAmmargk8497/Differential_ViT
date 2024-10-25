import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from DiffAttention import MultiheadDiffAttn  
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split
from DataLoader import CustomImageDataset
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from ViT import VisionTransformer  


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on Custom ImageNet")
    parser.add_argument('--data-dir', type=str, default= '/home/perioguatex/Desktop/Exprim1/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/',
                        help='Path to dataset directory structured as /train and /val with class subdirectories')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume from a checkpoint')
    parser.add_argument('--num-classes', type=int, default= 1000,required=True,
                        help='Number of output classes')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size for PatchEmbedding')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of Transformer blocks')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--mlp-dim', type=int, default=3072,
                        help='Dimension of MLP in Transformer')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for training')
    args = parser.parse_args()
    return args


def get_data_loaders(data_dir, image_size, batch_size, num_workers, selected_classes, val_split=0.1):
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Load the dataset and filter only the selected classes
    all_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=None)
    selected_class_indices = [all_data.class_to_idx[cls] for cls in selected_classes]
    filtered_samples = [sample for sample in all_data.samples if sample[1] in selected_class_indices]
    # Existing dataset code...
    num_val = int(len(filtered_samples) * val_split)
    val_samples = filtered_samples[:num_val]
    train_samples = filtered_samples[num_val:]

    # Create custom datasets
    train_dataset = CustomImageDataset(train_samples, transform=transform_train)
    val_dataset = CustomImageDataset(val_samples, transform=transform_val)

    # Create DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler, val_sampler


def initialize_model(args):
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        channels=args.channels,
        dropout=args.dropout,
    )
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, rank, scheduler=None, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if rank == 0:
        progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    else:
        progress_bar = data_loader

    for inputs, targets in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if rank == 0:
            progress_bar.set_postfix({'Loss': f'{running_loss / total:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    if scheduler:
        scheduler.step()

    return epoch_loss, epoch_acc


def validate(model, criterion, data_loader, device, epoch, rank):
    torch.cuda.empty_cache()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if rank == 0:
        progress_bar = tqdm(data_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
    else:
        progress_bar = data_loader

    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            if rank == 0:
                progress_bar.set_postfix({'Loss': f'{running_loss / total:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save checkpoint
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth'))


def main():
    args = parse_args()
    selected_classes = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Get local rank and set device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    print(f"Using device: {device}")

    # Get global rank
    rank = dist.get_rank()

    # Prepare data loaders
    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        selected_classes=selected_classes
    )

    # Initialize the model
    model = initialize_model(args)
    model = model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Learning rate scheduler (optional)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Optionally resume from a checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            if rank == 0:
                print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            if rank == 0:
                print(f"No checkpoint found at '{args.resume}'")

    # Create output directory with timestamp (only on main process)
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, rank,
            scheduler=scheduler, scaler=scaler
        )

        val_loss, val_acc = validate(model, criterion, val_loader, device, epoch, rank)

        # Reduce metrics across processes
        total_train_loss = torch.tensor(train_loss).to(device)
        total_train_acc = torch.tensor(train_acc).to(device)
        total_val_loss = torch.tensor(val_loss).to(device)
        total_val_acc = torch.tensor(val_acc).to(device)

        dist.reduce(total_train_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_train_acc, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_val_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_val_acc, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            total_train_loss /= dist.get_world_size()
            total_train_acc /= dist.get_world_size()
            total_val_loss /= dist.get_world_size()
            total_val_acc /= dist.get_world_size()

            print(f"Training   - Loss: {total_train_loss:.4f}, Acc: {total_train_acc:.2f}%")
            print(f"Validation - Loss: {total_val_loss:.4f}, Acc: {total_val_acc:.2f}%")

            # Check if this is the best model so far
            is_best = total_val_acc > best_acc
            best_acc = max(total_val_acc, best_acc)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            save_checkpoint(checkpoint, is_best, checkpoint_dir=output_dir, filename=f'epoch_{epoch + 1}.pth')

            if is_best:
                print(f"New best model with accuracy: {best_acc:.2f}%. Saved to '{output_dir}/best_model.pth'")

    if rank == 0:
        print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
        print(f"Checkpoints are saved in '{output_dir}'")


if __name__ == '__main__':
    main()
