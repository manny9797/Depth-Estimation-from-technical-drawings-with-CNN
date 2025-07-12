import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import DepthEstimationNet  # Custom model
from loss import CombinedLoss  # Custom loss function
from preprocessing import getTrainingData, getTestingData  # Data loaders
import gdown
import zipfile


def download_and_extract_dataset():
    """
    Downloads and extracts the dataset from Google Drive.
    """
    file_id = "1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "dataset.zip", quiet=False)
    with zipfile.ZipFile("./dataset.zip", "r") as zip_ref:
        zip_ref.extractall("dataset")
    print("✅ Dataset downloaded and extracted successfully.")


def train(args):
    """
    Main training function.
    Args:
        args: Parsed command-line arguments.
    """
    # Download and extract the dataset
    download_and_extract_dataset()

    # Initialize the model
    model = DepthEstimationNet(pretrained=args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint loaded from {args.checkpoint}")

    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Combined loss function with user-defined weights
    criterion = CombinedLoss(
        ssim_weight=args.loss_weights[0],
        berhu_weight=args.loss_weights[1],
        gradient_weight=args.loss_weights[2]
    )

    best_val_loss = float('inf')
    patience = 0
    max_patience = 5

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        training_data = getTrainingData(batch_size=10)
        total_batches = len(training_data)

        # Initialize epoch losses
        epoch_losses = {'total': 0.0, 'ssim': 0.0, 'berhu': 0.0, 'gradient': 0.0}

        for batch_idx, batch in enumerate(training_data):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Resize outputs to match target dimensions
            outputs_resized = F.interpolate(outputs, size=(114, 152), mode='bilinear', align_corners=False)

            # Compute loss
            loss, loss_components = criterion(outputs_resized, depths)
            epoch_losses['total'] += loss.item()
            epoch_losses['ssim'] += loss_components['ssim']
            epoch_losses['berhu'] += loss_components['berhu']
            epoch_losses['gradient'] += loss_components['gradient']

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1} - Batch {batch_idx+1}/{total_batches}")
                print(f"Loss: {loss:.4f}")

            # Backward pass and optimizer step
            loss.backward()
            # Apply gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Compute average losses for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= total_batches

        print(f"\n✅ Epoch {epoch+1} completed")
        print(f"📉 Training Losses:")
        print(f"   Total   : {epoch_losses['total']:.4f}")
        print(f"   SSIM    : {epoch_losses['ssim']:.4f}")
        print(f"   BerHu   : {epoch_losses['berhu']:.4f}")
        print(f"   Gradient: {epoch_losses['gradient']:.4f}")

        # Validation phase
        model.eval()
        val_losses = {'total': 0.0, 'ssim': 0.0, 'berhu': 0.0, 'gradient': 0.0}
        test_data = getTestingData(batch_size=10)
        total_test_batches = len(test_data)

        with torch.no_grad():
            for batch in test_data:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)

                # Forward pass and loss computation
                outputs = model(images)
                outputs_resized = F.interpolate(outputs, size=(114, 152), mode='bilinear', align_corners=False)

                loss, loss_components = criterion(outputs_resized, depths)
                val_losses['total'] += loss.item()
                val_losses['ssim'] += loss_components['ssim']
                val_losses['berhu'] += loss_components['berhu']
                val_losses['gradient'] += loss_components['gradient']

        # Compute average validation losses
        for key in val_losses:
            val_losses[key] /= total_test_batches

        print(f"🧪 Validation Losses:")
        print(f"   Total   : {val_losses['total']:.4f}")
        print(f"   SSIM    : {val_losses['ssim']:.4f}")
        print(f"   BerHu   : {val_losses['berhu']:.4f}")
        print(f"   Gradient: {val_losses['gradient']:.4f}")

        # Adjust learning rate based on validation loss
        scheduler.step(val_losses['total'])

        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model2.pth')
            print("✅ Best model saved.")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"\n✅ Epoch {epoch+1} completed")
        print(f"📉 Training - Total: {epoch_losses['total']:.4f}")
        print(f"🧪 Validation - Total: {val_losses['total']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation Training Script")
    # Whether to use pretrained weights
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model weights.")
    # Optional checkpoint path
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint (optional).", default=None)
    # Loss weights for CombinedLoss
    parser.add_argument("--loss_weights", type=float, nargs=3, metavar=("SSIM", "BERHU", "GRADIENT"),
                        default=[0.5, 0.5, 0], help="Loss weights for SSIM, BerHu, and Gradient components.")
    # Number of training epochs
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    train(args)