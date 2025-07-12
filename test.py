import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import DepthEstimationNet  # Custom model

# ImageNet statistics for normalization
__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def load_model(pretrained, checkpoint_path, device):
    """
    Load the depth estimation model with optional pretrained weights and checkpoint.
    
    Args:
        pretrained (bool): Whether to use pretrained weights for the model.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device to load the model onto.
    
    Returns:
        model (torch.nn.Module): The loaded model in evaluation mode.
    """
    # Initialize the model
    model = DepthEstimationNet(pretrained=pretrained)
    model.to(device)  # Move model to the specified device

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint loaded from {checkpoint_path}")

    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, device):
    """
    Load and preprocess the input image.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): The device to move the input tensor to.

    Returns:
        input_tensor (torch.Tensor): Preprocessed image tensor ready for inference.
    """
    # Load and convert the image to RGB
    image = Image.open(image_path).convert("RGB")

    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
    ])

    # Apply transformations and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor


def visualize_depth_map(depth_map):
    """
    Visualize the depth map using Matplotlib.

    Args:
        depth_map (np.ndarray): The depth map to visualize.
    """
    plt.imshow(depth_map, cmap="plasma")
    plt.colorbar()
    plt.show()


def main(args):
    """
    Main function to run depth estimation on an input image.

    Args:
        args: Parsed command-line arguments.
    """
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(pretrained=args.pretrained, checkpoint_path=args.checkpoint, device=device)

    # Preprocess the image
    input_tensor = preprocess_image(image_path=args.image_path, device=device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Convert the output to a NumPy array for visualization
    depth_map = output.squeeze().cpu().numpy()

    # Visualize the depth map
    visualize_depth_map(depth_map)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Depth Estimation Test Script")
    # Whether to use pretrained weights
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model weights.")
    # Path to the model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    # Path to the input image
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Run the main function
    main(args)