"""."""

# 1. Importing Libraries
import os
import warnings

import albumentations as A  # Library for data augmentation
import cv2
import matplotlib.pyplot as plt  # Used for plotting and visualizing data
import numpy as np  # Provides support for numerical operations on arrays
import pandas as pd  # Data manipulation and analysis library
import segmentation_models_pytorch as smp  # Contains segmentation models for PyTorch
import torch  # Main PyTorch library for deep learning
import torch.nn as nn  # Contains neural network layers and functions
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2  # Converts image data to PyTorch tensors
from sklearn.model_selection import (
    train_test_split,
)  # Splits data into train and test sets
from torch.optim import SGD, Adam, AdamW  # Adam optimizer for training neural networks
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)  # Scheduler to reduce learning rate on plateau
from torch.utils.data import (
    DataLoader,  # Helps manage data and create batches
    Dataset,
)
from torchinfo import summary  # Provides model summary similar to Keras

from lun_segmentation import data
from unet import model

warnings.filterwarnings("ignore")  # Suppresses warning messages

IN_CHANNELS = 3  # Number of input channels (e.g., grayscale images)
N_CLASSES = 3  # Number of output channels (e.g., binary segmentation)
TRAIN = False


def visualize_sample(df, index):
    """
    Visualizes a sample image and its corresponding mask.

    Parameters:
        df (DataFrame): The dataframe containing image and mask file names.
        index (int): The index of the sample to visualize.
    """
    # Get the image and mask file names from the DataFrame
    # random_row = df.sample(n=1).iloc[0]  # Get a random row from the DataFrame
    random_row = df.iloc[index]  # Get the row at the specified index
    img_name = random_row["ImageId"]
    mask_name = random_row["MaskId"]

    print(f"Index: {idx + 1} - Image: {img_name}, Mask: {mask_name}")  # Print the image and mask file names

    # Construct the file paths for the image and mask
    img_path = os.path.join("/workspace/data/chest_ct_segmentation/images/images", img_name)
    mask_path = os.path.join("/workspace/data/chest_ct_segmentation/masks/masks", mask_name)

    # Load the image and mask using OpenCV
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    # Preprocess the mask: threshold values to binary for each class
    # mask[mask < 240] = 0  # Set pixels below 240 to 0
    # mask[mask >= 240] = 1  # Set pixels 240 and above to 1

    # Plot the original image and the image with mask overlay
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axs[0].imshow(image)  # Convert from BGR to RGB color space
    axs[0].set_title("Original Image")  # Set title for the original image
    axs[0].axis("off")  # Hide axes for a cleaner look

    # Display the image with overlayed masks for each class
    axs[1].imshow(image)  # Display the original image as the background
    axs[1].imshow(mask[:, :, 0], alpha=0.3, cmap="Reds")  # Overlay lung mask with transparency
    axs[1].imshow(mask[:, :, 1], alpha=0.3, cmap="Blues")  # Overlay heart mask with transparency
    axs[1].imshow(mask[:, :, 2], alpha=0.3, cmap="Greens")  # Overlay trachea mask with transparency
    axs[1].set_title("Image with Mask Overlay")  # Set title for the overlay image
    axs[1].axis("off")  # Hide axes for a cleaner look

    # Display the plot
    plt.show()


# Visualize the first 3 samples
# for idx in range(4000):  # Loop through the first three indices
#     df = pd.read_csv("/workspace/data/chest_ct_segmentation/train.csv")
#     visualize_sample(df, idx)  # Call the visualize_sample function to display each image and its mask


class ChestCTDataset(Dataset):
    """Custom Dataset for loading Chest CT images and masks."""

    def __init__(self, df, images_dir, masks_dir, transform=None):
        """
        Initialization.

        Parameters:
            df (DataFrame): DataFrame containing image and mask file names.
            images_dir (str): Directory path where images are stored.
            masks_dir (str): Directory path where masks are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)  # Reset index of DataFrame to ensure ordered indices
        self.images_dir = images_dir  # Directory containing images
        self.masks_dir = masks_dir  # Directory containing masks
        self.transform = transform  # Transformations to be applied to the image and mask

    def __len__(self):
        """Return the total number of samples."""
        return len(self.df)  # Returns the number of rows in the DataFrame

    def __getitem__(self, idx):
        """
        Generate one sample of data.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): Transformed image tensor.
            mask (Tensor): Transformed mask tensor.
        """
        # Retrieve the image and mask file names for the given index
        img_name = self.df.loc[idx, 'ImageId']
        mask_name = self.df.loc[idx, 'MaskId']

        # Load image and mask using OpenCV
        img_path = os.path.join(self.images_dir, img_name)  # Full path to the image
        mask_path = os.path.join(self.masks_dir, mask_name)  # Full path to the mask
        image = cv2.imread(img_path)  # Load the image
        mask = cv2.imread(mask_path)  # Load the mask

        # Preprocess the mask: set values below 240 to 0 and values 240 or above to 1
        mask[mask < 240] = 0
        mask[mask >= 240] = 1

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)  # Apply transformations
            image = augmented['image']  # Transformed image
            mask = augmented['mask']  # Transformed mask

        # Change mask dimensions to [channels, height, width] for compatibility with PyTorch
        mask = mask.permute(2, 0, 1).float()  # Reorder dimensions and convert to float

        return image, mask  # Return the processed image and mask


# 4. Dataset and DataLoader
class Config:
    """Configuration parameters for the training process."""

    seed = 42  # Random seed for reproducibility
    data_csv = (
        "/workspace/data/chest_ct_segmentation/train_filtered.csv"  # Path to the CSV file containing image and mask file names
    )
    images_dir = "/workspace/data/chest_ct_segmentation/images/images"  # Directory containing images
    masks_dir = "/workspace/data/chest_ct_segmentation/masks/masks"  # Directory containing masks
    batch_size = 4  # Number of samples per batch
    num_workers = 2  # Number of workers for data loading
    num_epochs = 10  # Number of training epochs # STUDENTS WILL CHANGE THE NUM OF EPOCHS TO 2
    lr = 1e-4  # Learning rate for optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    val_size = 0.2  # Proportion of data for validation set
    test_size = 0.2  # Proportion of data for test set
    accumulation_steps = 32  # Steps for gradient accumulation to stabilize training


def seed_everything(seed):
    """Set seed for reproducibility in random number generation."""
    np.random.seed(seed)  # Set seed for numpy random operations
    torch.manual_seed(seed)  # Set seed for torch operations on CPU
    torch.cuda.manual_seed_all(seed)  # Set seed for torch operations on GPU


def prepare_dataloader(df, images_dir, masks_dir, phase):
    """
    Creates and returns a DataLoader for the given dataset.

    Parameters:
        df (DataFrame): DataFrame containing image and mask file names.
        images_dir (str): Directory path where images are stored.
        masks_dir (str): Directory path where masks are stored.
        phase (str): 'train' or 'val' to determine the type of DataLoader.

    Returns:
        dataloader (DataLoader): PyTorch DataLoader for the dataset.
    """
    # Create a dataset instance with the specified transformations
    dataset = ChestCTDataset(
        df=df,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=get_transforms(phase)  # Apply transformations based on phase
    )

    # Create a DataLoader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,  # Batch size from configuration
        shuffle=True if phase == 'train' else False,  # Shuffle data if in training phase
        num_workers=config.num_workers,  # Number of workers for data loading
        pin_memory=True  # Keep data in pinned memory for faster GPU transfer
    )

    return dataloader  # Return the DataLoader instance


def get_transforms(phase):
    """
    Returns a composition of data augmentations based on the phase.

    Parameters:
        phase (str): 'train' or 'val' to determine the type of augmentations.

    Returns:
        transform (Compose): Albumentations Compose object with the transformations.
    """
    if phase == 'train':
        # Define augmentations for the training phase
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
            A.VerticalFlip(p=0.5),  # Apply vertical flip with 50% probability
            A.RandomRotate90(p=0.5),  # Rotate the image by 90 degrees randomly with 50% probability
            A.Normalize(mean=(0.485, 0.456, 0.406),  # Normalize the image with mean and std
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),  # Convert the image and mask to PyTorch tensors
        ])
    else:
        # Define augmentations for the validation phase (only normalization)
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),  # Normalize the image with mean and std
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),  # Convert the image and mask to PyTorch tensors
        ])


# Initialize configuration and set random seed
config = Config()  # Create an instance of the Config class with defined parameters
seed_everything(config.seed)  # Set the seed based on the configuration for reproducibility

# Read the CSV file
data_df = pd.read_csv(config.data_csv)  # Load the dataset information from the CSV file

# Split into training and test sets
train_df, test_df = train_test_split(
    data_df,
    test_size=config.test_size,  # Proportion for the test set from configuration
    random_state=config.seed  # Set seed for reproducibility
)

# Split into training and validation sets
train_df, val_df = train_test_split(
    train_df,
    test_size=config.val_size,  # Proportion for the validation set from configuration
    random_state=config.seed  # Set seed for reproducibility
)

print(train_df.head())  # Display the first few rows of the training DataFrame
print(val_df.head())  # Display the first few rows of the validation DataFrame
print(test_df.head())  # Display the first few rows of the test DataFrame

# Reset index after split to ensure continuity in DataFrame indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Prepare DataLoaders for each dataset split
train_loader = prepare_dataloader(train_df, config.images_dir, config.masks_dir, 'train')  # Training DataLoader
val_loader = prepare_dataloader(val_df, config.images_dir, config.masks_dir, 'val')        # Validation DataLoader
test_loader = prepare_dataloader(test_df, config.images_dir, config.masks_dir, 'test')     # Test DataLoader


class DiceLoss(nn.Module):
    """Dice Loss implementation for measuring the overlap between predictions and targets."""

    def __init__(self, eps=1e-7):
        """
        Initialize DiceLoss with a small epsilon to avoid division by zero.

        Parameters:
            eps (float): Small constant to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.eps = eps  # Epsilon value to ensure numerical stability

    def forward(self, outputs, targets):
        """
        Forward pass for Dice Loss computation.

        Parameters:
            outputs (Tensor): Model predictions before activation.
            targets (Tensor): Ground truth masks.

        Returns:
            loss (Tensor): Computed Dice Loss.
        """
        # Apply sigmoid activation to outputs to get probabilities
        probabilities = torch.sigmoid(outputs)

        # Flatten the tensors for simplified Dice calculation
        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Calculate intersection and union between probabilities and targets
        intersection = (probabilities * targets).sum(dim=1)
        union = probabilities.sum(dim=1) + targets.sum(dim=1)

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        # Return Dice Loss by subtracting the mean Dice coefficient from 1
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combination of Binary Cross-Entropy (BCE) Loss and Dice Loss for segmentation tasks."""

    def __init__(self):
        """
        Initialize BCEDiceLoss with BCE and Dice components.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # BCE loss for pixel-wise binary classification
        self.dice = DiceLoss()  # Dice loss for overlap measurement

    def forward(self, outputs, targets):
        """
        Forward pass for BCEDiceLoss.

        Parameters:
            outputs (Tensor): Model predictions before activation.
            targets (Tensor): Ground truth masks.

        Returns:
            loss (Tensor): Computed combined loss.
        """
        # Compute BCE loss between outputs and targets
        bce_loss = self.bce(outputs, targets)

        # Compute Dice loss between outputs and targets
        dice_loss = self.dice(outputs, targets)

        # Return the sum of BCE and Dice losses as the combined loss
        return bce_loss + dice_loss


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2D -> ReLU) x 2

    This block applies two consecutive convolutional layers,
    each followed by a ReLU activation function.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize DoubleConv block with two convolutional layers.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolution with kernel size 3
            nn.BatchNorm2d(out_channels),  # Batch normalization for stable training
            nn.ReLU(inplace=True),  # ReLU activation for non-linearity
            # Second convolutional layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Another convolution with kernel size 3
            nn.BatchNorm2d(out_channels),  # Batch normalization for stable training
            nn.ReLU(inplace=True),  # ReLU activation for non-linearity
        )

    def forward(self, x):
        """
        Forward pass through the DoubleConv block.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output after applying two convolutions with ReLU.
        """
        return self.double_conv(x)  # Apply the sequential layers to the input


class DownBlock(nn.Module):
    """
    Downsampling Block: MaxPool2d -> DoubleConv

    This block performs downsampling using MaxPooling,
    followed by a DoubleConv block to extract features.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize DownBlock with a max-pooling layer and a DoubleConv block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            # Max Pooling layer to downsample the input by a factor of 2
            nn.MaxPool2d(kernel_size=2),
            # Double convolutional layers to extract features
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        """
        Forward pass through the DownBlock.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output after max-pooling and DoubleConv layers.
        """
        return self.maxpool_conv(x)  # Apply max-pooling and double convolution sequentially


class UpBlock(nn.Module):
    """
    Upsampling Block: Upsample or ConvTranspose2d -> Concatenate -> DoubleConv

    This block performs upsampling using either bilinear interpolation
    or transposed convolution, concatenates the result with the corresponding
    feature map from the contracting path (skip connection), and applies
    a DoubleConv block.
    """

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        """
        Initialize UpBlock with upsampling and DoubleConv operations.

        Parameters:
            in_channels (int): Number of input channels.
            skip_channels (int): Number of channels in the skip connection.
            out_channels (int): Number of output channels after the block.
            bilinear (bool): Whether to use bilinear interpolation or transposed convolution.
        """
        super(UpBlock, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            # Use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # 1x1 Convolution to adjust channels after upsampling
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            # DoubleConv after concatenation of upsampled x1 and x2
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        else:
            # Use transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # DoubleConv after concatenation of upsampled x1 and x2
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass for the UpBlock.

        Parameters:
            x1 (Tensor): Output from the previous layer (decoder).
            x2 (Tensor): Corresponding feature map from the encoder (skip connection).

        Returns:
            Tensor: Output after upsampling, concatenation, and DoubleConv.
        """
        # Upsample x1
        x1 = self.up(x1)

        # Adjust channels if using bilinear upsampling
        if self.bilinear:
            x1 = self.conv1x1(x1)

        # Adjust padding to match dimensions of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,  # Pad x1 to match height and width of x2
                diffY // 2,
                diffY - diffY // 2,
            ],
        )

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)

        # Apply DoubleConv and return the result
        x = self.conv(x)
        return x


class OutBlock(nn.Module):
    """
    Output Block: Conv2D to get the desired number of classes.

    This block applies a 1x1 convolution to reduce the number
    of channels to the number of classes, producing the final
    segmentation map.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize OutBlock with a 1x1 convolution.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (number of classes).
        """
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution for channel reduction

    def forward(self, x):
        """
        Forward pass for OutBlock.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with the desired number of channels.
        """
        return self.conv(x)  # Apply 1x1 convolution to produce the final output


class UNet(nn.Module):
    """
    U-Net Model

    The U-Net is a convolutional network architecture for fast and
    precise segmentation of images.
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        Initialize the U-Net model.

        Parameters:
            n_channels (int): Number of input channels (e.g., 3 for RGB).
            n_classes (int): Number of output classes.
            bilinear (bool): Whether to use bilinear interpolation or transposed convolution.
        """
        super(UNet, self).__init__()

        self.n_channels = n_channels  # Number of input channels (e.g., 3 for RGB)
        self.n_classes = n_classes  # Number of output classes
        self.bilinear = bilinear  # Whether to use bilinear interpolation or transposed convolution

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)  # Initial convolution block
        self.down1 = DownBlock(64, 128)  # First downsampling block
        self.down2 = DownBlock(128, 256)  # Second downsampling block
        self.down3 = DownBlock(256, 512)  # Third downsampling block (bottleneck)

        # Decoder (Expanding Path)
        self.up1 = UpBlock(512, 256, 256, bilinear)  # First upsampling block
        self.up2 = UpBlock(256, 128, 128, bilinear)  # Second upsampling block
        self.up3 = UpBlock(128, 64, 64, bilinear)  # Third upsampling block

        # Output Layer
        self.outc = OutBlock(64, n_classes)  # Final output layer with class prediction

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Parameters:
            x (Tensor): Input tensor with shape [batch_size, n_channels, H, W].

        Returns:
            logits (Tensor): Output tensor with shape [batch_size, n_classes, H, W].
        """
        # Encoder path
        x1 = self.inc(x)  # Initial conv block, output: [batch_size, 64, H, W]
        x2 = self.down1(x1)  # First down block, output: [batch_size, 128, H/2, W/2]
        x3 = self.down2(x2)  # Second down block, output: [batch_size, 256, H/4, W/4]
        x4 = self.down3(x3)  # Third down block, output: [batch_size, 512, H/8, W/8]

        # Decoder path with skip connections
        x = self.up1(x4, x3)  # First up block, input x4 and skip connection x3
        x = self.up2(x, x2)  # Second up block, input from previous layer and skip connection x2
        x = self.up3(x, x1)  # Third up block, input from previous layer and skip connection x1

        # Output layer
        logits = self.outc(x)  # Final 1x1 conv to get class logits, output: [batch_size, n_classes, H, W]
        return logits


def train_one_epoch(model, loader, criterion, optimizer):
    """
    Train the model for one epoch.

    Parameters:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        epoch_loss (float): Average loss for the epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch

    # Iterate over the batches in the DataLoader
    for images, masks in loader:
        images = images.to(config.device)  # Move images to the configured device (CPU/GPU)
        masks = masks.to(config.device)  # Move masks to the configured device

        optimizer.zero_grad()  # Reset gradients from previous batch
        outputs = model(images)  # Forward pass to get model predictions
        loss = criterion(outputs, masks)  # Calculate the loss
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update model parameters

        running_loss += loss.item() * images.size(0)  # Accumulate batch loss scaled by batch size

    epoch_loss = running_loss / len(loader.dataset)  # Calculate average loss for the epoch
    return epoch_loss  # Return the average epoch loss


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Calculate Dice Coefficient and Intersection over Union (IoU) for model predictions.

    Parameters:
        outputs (Tensor): Model predictions before activation.
        targets (Tensor): Ground truth masks.
        threshold (float): Threshold to binarize predictions.

    Returns:
        dice (float): Dice Coefficient.
        iou (float): Intersection over Union (Jaccard Index).
    """
    # Apply sigmoid activation to the outputs and apply threshold to obtain binary predictions
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()  # Binarize predictions based on threshold

    # Flatten the tensors for metric calculation
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Calculate intersection and union for Dice coefficient
    intersection = (preds * targets).sum(dim=1)  # Overlapping area between prediction and target
    union = preds.sum(dim=1) + targets.sum(dim=1)  # Sum of prediction and target areas
    dice = (2.0 * intersection) / (union + 1e-7)  # Compute Dice coefficient

    # Calculate intersection and union for IoU
    intersection_iou = (preds * targets).sum(dim=1)  # Overlapping area for IoU
    union_iou = (preds.sum(dim=1) + targets.sum(dim=1)) - intersection_iou  # Union area for IoU
    iou = (intersection_iou) / (union_iou + 1e-7)  # Compute IoU

    # Return the average Dice and IoU metrics
    return dice.mean().item(), iou.mean().item()


def validate_model(model, loader, criterion):
    """
    Validate the model.

    Parameters:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.

    Returns:
        epoch_loss (float): Average loss for the epoch.
        epoch_dice (float): Average Dice Coefficient.
        epoch_iou (float): Average IoU.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0  # Initialize running loss for the epoch
    total_dice = 0.0  # Initialize total Dice score for the epoch
    total_iou = 0.0  # Initialize total IoU score for the epoch

    # Disable gradient calculation for validation to save memory
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(config.device)  # Move images to the configured device (CPU/GPU)
            masks = masks.to(config.device)  # Move masks to the configured device

            outputs = model(images)  # Forward pass to get model predictions
            loss = criterion(outputs, masks)  # Calculate the loss
            running_loss += loss.item() * images.size(0)  # Accumulate batch loss scaled by batch size

            # Calculate Dice and IoU metrics
            dice, iou = calculate_metrics(outputs, masks)
            total_dice += dice * images.size(0)  # Accumulate Dice score scaled by batch size
            total_iou += iou * images.size(0)  # Accumulate IoU score scaled by batch size

    # Calculate average loss and metrics for the epoch
    epoch_loss = running_loss / len(loader.dataset)  # Average loss for the epoch
    epoch_dice = total_dice / len(loader.dataset)  # Average Dice score for the epoch
    epoch_iou = total_iou / len(loader.dataset)  # Average IoU score for the epoch

    return epoch_loss, epoch_dice, epoch_iou  # Return loss, Dice, and IoU for the epoch


def calculate_metrics_per_class(outputs, targets, threshold=0.5):
    """
    Calculate Dice Coefficient and IoU for each class (channel).

    Parameters:
        outputs (Tensor): Model predictions before activation.
        targets (Tensor): Ground truth masks.
        threshold (float): Threshold to binarize predictions.

    Returns:
        dice_per_class (Tensor): Dice Coefficient for each class (channel).
        iou_per_class (Tensor): Intersection over Union (IoU) for each class (channel).
    """
    # Apply sigmoid activation and threshold
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    # Flatten the tensors by channel
    preds = preds.view(preds.size(0), preds.size(1), -1)  # Shape: [batch_size, channels, height*width]
    targets = targets.view(targets.size(0), targets.size(1), -1)  # Shape: [batch_size, channels, height*width]

    # Initialize lists to store Dice and IoU for each channel
    dice_per_class = []
    iou_per_class = []

    # Calculate Dice and IoU for each class (channel)
    for c in range(preds.size(1)):  # Iterate over each channel (class)
        # Calculate intersection and union for Dice
        intersection = (preds[:, c] * targets[:, c]).sum(dim=1)  # Intersection for class c
        union = preds[:, c].sum(dim=1) + targets[:, c].sum(dim=1)  # Union for class c
        dice = (2.0 * intersection) / (union + 1e-7)  # Dice coefficient for class c
        dice_per_class.append(dice * 100)

        # Calculate intersection and union for IoU
        intersection_iou = (preds[:, c] * targets[:, c]).sum(dim=1)  # Intersection for class c
        union_iou = (preds[:, c].sum(dim=1) + targets[:, c].sum(dim=1)) - intersection_iou  # Union for class c
        iou = intersection_iou / (union_iou + 1e-7)  # IoU for class c
        iou_per_class.append(iou * 100)

    # Stack results into tensors for each class (channel)
    dice_per_class = torch.stack(dice_per_class, dim=0).mean(dim=1)  # Average Dice per batch for each class
    iou_per_class = torch.stack(iou_per_class, dim=0).mean(dim=1)  # Average IoU per batch for each class

    return dice_per_class, iou_per_class  # Return Dice and IoU for each class


def validate_model_per_class(model, loader, criterion):
    """
    Validate the model.

    Parameters:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.

    Returns:
        epoch_loss (float): Average loss for the epoch.
        epoch_dice (float): Average Dice Coefficient.
        epoch_iou (float): Average IoU.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0  # Initialize running loss
    total_dice = 0.0  # Initialize running total for Dice metric
    total_iou = 0.0  # Initialize running total for IoU metric

    # Disable gradient calculations for validation
    with torch.no_grad():
        # Loop through the validation data loader
        for images, masks in loader:
            images = images.to(config.device)  # Move images to the device (GPU/CPU)
            masks = masks.to(config.device)  # Move masks to the device

            outputs = model(images)  # Generate predictions from the model
            loss = criterion(outputs, masks)  # Calculate the loss
            running_loss += loss.item() * images.size(0)  # Accumulate the batch loss

            # Calculate Dice and IoU metrics per class
            dice, iou = calculate_metrics_per_class(outputs, masks)
            total_dice += dice * images.size(0)  # Accumulate Dice score per batch
            total_iou += iou * images.size(0)  # Accumulate IoU score per batch

    # Calculate average loss, Dice, and IoU for the epoch
    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = total_dice / len(loader.dataset)
    epoch_iou = total_iou / len(loader.dataset)

    return epoch_loss, epoch_dice, epoch_iou  # Return metrics for the validation epoch


# Define the loss function
criterion = BCEDiceLoss()  # Combined BCE and Dice loss for segmentation


if TRAIN:
    # Initialize U-Net model with EfficientNet-B2 encoder
    model_efficientUnet = smp.Unet(
        encoder_name="efficientnet-b2",  # Encoder architecture
        encoder_weights="imagenet",  # Use pre-trained weights for encoder
        in_channels=IN_CHANNELS,  # Number of input channels (RGB)
        classes=N_CLASSES,  # Number of output channels (number of classes in segmentation task)
        activation=None,  # No activation function at the final layer
    )

    # Display a summary of the model, including the input size and number of parameters
    summary(model_efficientUnet, input_size=(config.batch_size, IN_CHANNELS, 512, 512))

    # Move model to device (GPU or CPU)
    model_efficientUnet = model_efficientUnet.to(config.device)  # Transfer model to the configured device

    # Define the optimizer
    optimizer_efficientUnet = Adam(
        model_efficientUnet.parameters(), lr=config.lr
    )  # Adam optimizer with learning rate from config

    # Define the learning rate scheduler
    scheduler_efficientUnet = ReduceLROnPlateau(
        optimizer_efficientUnet,  # Optimizer to adjust
        mode="min",  # Reduce LR when monitored metric has stopped decreasing
        patience=3,  # Number of epochs to wait before reducing LR
        # verbose=True,  # Print message when learning rate is reduced
    )
    # Initialize variables to track the best model
    best_val_loss = float("inf")  # Set best validation loss to a high initial value

    # Lists to store loss and metrics for each epoch
    train_losses_efficientUnet = []  # List to track training losses
    val_losses_efficientUnet = []  # List to track validation losses
    dice_scores_efficientUnet = []  # List to track Dice scores for each epoch
    iou_scores_efficientUnet = []  # List to track IoU scores for each epoch

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print("-" * 30)

        # Training phase
        train_loss = train_one_epoch(model_efficientUnet, train_loader, criterion, optimizer_efficientUnet)
        train_losses_efficientUnet.append(train_loss)  # Store training loss for the epoch
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss, val_dice, val_iou = validate_model(model_efficientUnet, val_loader, criterion)
        val_losses_efficientUnet.append(val_loss)  # Store validation loss for the epoch
        dice_scores_efficientUnet.append(val_dice)  # Store validation Dice score
        iou_scores_efficientUnet.append(val_iou)  # Store validation IoU score
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        # Step the scheduler based on validation loss
        scheduler_efficientUnet.step(val_loss)

        # Check if this is the best model so far, and save it if so
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model_efficientUnet.state_dict(),
                "/root/models/best_model_efficient_Unet.pth",
            )
            print("Saved Best Model")

        print("-" * 30)

    # Release GPU memory
    del model_efficientUnet  # Delete the model to free up memory
    torch.cuda.empty_cache()  # Clear unused cached memory on the GPU

    # Initialize a custom U-Net model with the specified input and output channels
    model_unet = model.UNet(
        in_channels=IN_CHANNELS, n_classes=N_CLASSES
    )  # U-Net with 3 input channels (RGB) and 3 output classes
    model_unet = model_unet.to(config.device)  # Move the model to the configured device (CPU/GPU)

    # Display a summary of the model, including input size and number of parameters
    summary(model_unet, input_size=(config.batch_size, IN_CHANNELS, 512, 512))

    # Define the optimizer for the custom U-Net model
    optimizer_unet = Adam(model_unet.parameters(), lr=config.lr)  # Adam optimizer with learning rate from config

    # Define the learning rate scheduler
    scheduler_unet = ReduceLROnPlateau(
        optimizer_unet,  # Optimizer to adjust
        mode="min",  # Reduce LR when the monitored metric has stopped decreasing
        patience=3,  # Number of epochs to wait before reducing LR
        # verbose=True,  # Print a message when the learning rate is reduced
    )

    # Initialize variables to track the best model
    best_val_loss = float("inf")  # Set initial best validation loss to a high value

    # Lists to store loss and metrics for each epoch
    train_losses_unet = []  # List to track training losses
    val_losses_unet = []  # List to track validation losses
    dice_scores_unet = []  # List to track Dice scores for each epoch
    iou_scores_unet = []  # List to track IoU scores for each epoch

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print("-" * 30)

        # Training phase
        train_loss = train_one_epoch(model_unet, train_loader, criterion, optimizer_unet)
        train_losses_unet.append(train_loss)  # Store training loss for the epoch
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss, val_dice, val_iou = validate_model(model_unet, val_loader, criterion)
        val_losses_unet.append(val_loss)  # Store validation loss for the epoch
        dice_scores_unet.append(val_dice)  # Store validation Dice score
        iou_scores_unet.append(val_iou)  # Store validation IoU score
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        # Step the scheduler based on validation loss
        scheduler_unet.step(val_loss)

        # Check if this is the best model so far and save it if so
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model_unet.state_dict(),
                "/root/models/best_model_unet.pth"
            )
            print("Saved Best Model")

        print("-" * 30)

    # Release GPU memory
    del model_unet  # Delete the U-Net model with transposed convolution to free up memory
    torch.cuda.empty_cache()  # Clear unused cached memory on the GPU

    # Initialize a custom U-Net model with transposed convolution (no bilinear interpolation)
    model_unet_nb = UNet(
        n_channels=IN_CHANNELS, n_classes=N_CLASSES, bilinear=False
    )  # U-Net with 3 input channels and 3 output classes
    model_unet_nb = model_unet_nb.to(config.device)  # Move the model to the configured device (CPU/GPU)

    # Display a summary of the model, including input size and number of parameters
    summary(model_unet_nb, input_size=(config.batch_size, IN_CHANNELS, 512, 512))

    # Define the optimizer for the custom U-Net model
    optimizer_unet_nb = Adam(model_unet_nb.parameters(), lr=config.lr)  # Adam optimizer with learning rate from config

    # Define the learning rate scheduler
    scheduler_unet_nb = ReduceLROnPlateau(
        optimizer_unet_nb,  # Optimizer to adjust
        mode="min",  # Reduce LR when the monitored metric has stopped decreasing
        patience=3,  # Number of epochs to wait before reducing LR
        # verbose=True,  # Print a message when the learning rate is reduced
    )

    # Initialize variables to track the best model
    best_val_loss = float("inf")  # Set initial best validation loss to a high value

    # Lists to store loss and metrics for each epoch
    train_losses_unet_nb = []  # List to track training losses
    val_losses_unet_nb = []  # List to track validation losses
    dice_scores_unet_nb = []  # List to track Dice scores for each epoch
    iou_scores_unet_nb = []  # List to track IoU scores for each epoch

    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print("-" * 30)

        # Training phase
        train_loss = train_one_epoch(model_unet_nb, train_loader, criterion, optimizer_unet_nb)
        train_losses_unet_nb.append(train_loss)  # Store training loss for the epoch
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss, val_dice, val_iou = validate_model(model_unet_nb, val_loader, criterion)
        val_losses_unet_nb.append(val_loss)  # Store validation loss for the epoch
        dice_scores_unet_nb.append(val_dice)  # Store validation Dice score
        iou_scores_unet_nb.append(val_iou)  # Store validation IoU score
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        # Step the scheduler based on validation loss
        scheduler_unet_nb.step(val_loss)

        # Check if this is the best model so far and save it if so
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model_unet_nb.state_dict(),
                "/root/models/best_model_unet_nb.pth",
            )
            print("Saved Best Model")

        print("-" * 30)

    # Release GPU memory
    del model_unet_nb  # Delete the U-Net model with transposed convolution to free up memory
    torch.cuda.empty_cache()  # Clear unused cached memory on the GPU

# Initialize U-Net model with EfficientNet-B2 encoder
model_efficientUnet = smp.Unet(
    encoder_name="efficientnet-b2",  # Encoder architecture
    encoder_weights="imagenet",  # Use pre-trained weights for encoder
    in_channels=IN_CHANNELS,  # Number of input channels (RGB)
    classes=N_CLASSES,  # Number of output channels (classes)
    activation=None,  # No activation at the end
)
model_efficientUnet = model_efficientUnet.to(config.device)  # Move model to configured device (CPU/GPU)

# Initialize a custom U-Net model
model_unet = model.UNet(
    in_channels=IN_CHANNELS, n_classes=N_CLASSES
)  # U-Net with 3 input channels and 3 output classes
model_unet = model_unet.to(config.device)  # Move model to configured device (CPU/GPU)

# Initialize a custom U-Net model with transposed convolution instead of bilinear interpolation
model_unet_nb = UNet(n_channels=IN_CHANNELS, n_classes=N_CLASSES, bilinear=False)  # U-Net with transposed convolution
model_unet_nb = model_unet_nb.to(config.device)  # Move model to configured device (CPU/GPU)

# Load the best saved weights for the U-Net model with EfficientNet-B2 encoder
model_efficientUnet.load_state_dict(torch.load("/root/models/best_model_efficient_Unet.pth"))
model_efficientUnet.eval()  # Set the model to evaluation mode

# Load the best saved weights for the custom U-Net model
model_unet.load_state_dict(torch.load("/root/models/best_model_unet.pth"))
model_unet.eval()  # Set the model to evaluation mode

# Load the best saved weights for the custom U-Net model with transposed convolution
model_unet_nb.load_state_dict(torch.load("/root/models/best_model_unet_nb.pth"))
model_unet_nb.eval()  # Set the model to evaluation mode

# Final evaluation on the validation set for the EfficientNet-based U-Net model
final_val_loss, final_val_dice, final_val_iou = validate_model(model_efficientUnet, val_loader, criterion)

# Print final validation metrics
print(f"Final Validation Loss EfficientUnet: {final_val_loss:.4f}")
print(f"Final Validation Dice Coefficient EfficientUnet: {final_val_dice:.4f}")
print(f"Final Validation Jaccard Index EfficientUnet: {final_val_iou:.4f}")
print("-" * 30)

# Final evaluation on the validation set for the custom U-Net model
final_val_loss, final_val_dice, final_val_iou = validate_model(model_unet, val_loader, criterion)

# Print final validation metrics
print(f"Final Validation Loss Unet: {final_val_loss:.4f}")
print(f"Final Validation Dice Coefficient Unet: {final_val_dice:.4f}")
print(f"Final Validation Jaccard Index Unet: {final_val_iou:.4f}")
print("-" * 30)

# Final evaluation on the validation set for the custom U-Net model with transposed convolution (deconv)
final_val_loss, final_val_dice, final_val_iou = validate_model(model_unet_nb, val_loader, criterion)

# Print final validation metrics
print(f"Final Validation Loss Unet notebook: {final_val_loss:.4f}")
print(f"Final Validation Dice Coefficient Unet notebook: {final_val_dice:.4f}")
print(f"Final Validation Jaccard Index Unet notebook: {final_val_iou:.4f}")
print("-" * 30)

if False:
    # Plot the training and validation loss for each model over epochs
    plt.figure(figsize=(10, 5))

    # Plot losses for the EfficientNet-based U-Net model
    plt.plot(train_losses_efficientUnet, label="Training Loss EfficientUnet", marker="o")
    plt.plot(val_losses_efficientUnet, label="Validation Loss EfficientUnet", marker="o")

    # Plot losses for the custom U-Net model
    plt.plot(train_losses_unet, label="Training Loss Unet", marker="o")
    plt.plot(val_losses_unet, label="Validation Loss Unet", marker="o")

    # Plot losses for the U-Net model with transposed convolution
    plt.plot(train_losses_unet_nb, label="Training Loss Unet notebook", marker="o")
    plt.plot(val_losses_unet_nb, label="Validation Loss Unet notebook", marker="o")

    # Add title, labels, legend, and grid
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(
    #     f"/workspace/code/lung-segmentation-tfm/resources/models/best_model_efficient_Unet_history.png",
    #     bbox_inches="tight",
    # )

    # Plot the Dice Coefficient over epochs for each model
    plt.figure(figsize=(10, 5))
    plt.plot(
        dice_scores_efficientUnet,
        label="Validation Dice EfficientUnet",
        color="green",
        marker="o",
    )
    plt.plot(dice_scores_unet, label="Validation Dice Unet", color="blue", marker="o")
    plt.plot(
        dice_scores_unet_nb,
        label="Validation Dice Unet notebook",
        color="red",
        marker="o",
    )
    plt.title("Validation Dice Coefficient Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(
    #     f"/workspace/code/lung-segmentation-tfm/resources/models/best_model_unet_history.png",
    #     bbox_inches="tight",
    # )

    # Plot the Jaccard Index (IoU) over epochs for each model
    plt.figure(figsize=(10, 5))
    plt.plot(
        iou_scores_efficientUnet,
        label="Validation Jaccard Index EfficientUnet",
        color="green",
        marker="o",
    )
    plt.plot(iou_scores_unet, label="Validation Jaccard Index Unet", color="blue", marker="o")
    plt.plot(
        iou_scores_unet_nb,
        label="Validation Jaccard Index Unet notebook",
        color="red",
        marker="o",
    )
    plt.title("Validation Jaccard Index Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Jaccard Index (IoU)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(
    #     f"/workspace/code/lung-segmentation-tfm/resources/models/best_model_unet_nb_history.png",
    #     bbox_inches="tight",
    # )

# Final evaluation on the test set for the EfficientNet-based U-Net model
final_test_loss, final_test_dice, final_test_iou = validate_model(model_efficientUnet, test_loader, criterion)

# Print final test metrics
print(f"Final Test Loss EfficientUnet: {final_test_loss:.4f}")
print(f"Final Test Dice Coefficient EfficientUnet: {final_test_dice:.4f}")
print(f"Final Test Jaccard Index EfficientUnet: {final_test_iou:.4f}")
print("-" * 30)

# Final evaluation on the test set for the custom U-Net model
final_test_loss, final_test_dice, final_test_iou = validate_model(model_unet, test_loader, criterion)

# Print final test metrics
print(f"Final Test Loss Unet: {final_test_loss:.4f}")
print(f"Final Test Dice Coefficient Unet: {final_test_dice:.4f}")
print(f"Final Test Jaccard Index Unet: {final_test_iou:.4f}")
print("-" * 30)

# Final evaluation on the test set for the U-Net model with transposed convolution (deconv)
final_test_loss, final_test_dice, final_test_iou = validate_model(model_unet_nb, test_loader, criterion)

# Print final test metrics
print(f"Final Test Loss Unet notebook: {final_test_loss:.4f}")
print(f"Final Test Dice Coefficient Unet notebook: {final_test_dice:.4f}")
print(f"Final Test Jaccard Index Unet notebook: {final_test_iou:.4f}")
print("-" * 30)


def visualize_predictions(model, dataset, index, model_name: str):
    """
    Visualizes the model's prediction on a single sample with colored masks.

    Parameters:
        model (nn.Module): The trained segmentation model.
        dataset (Dataset): The dataset containing the sample.
        index (int): Index of the sample to visualize.
    """
    # Retrieve the image and mask from the dataset
    image, mask = dataset[index]

    # Convert image tensor to numpy array and denormalize
    # Assuming the image was normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)

    # If image is grayscale (single channel), convert to 3-channel RGB
    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)

    # Convert mask tensor to numpy array
    mask_np = mask.cpu().numpy()  # Shape: (C, H, W)

    # Ensure mask has three channels for Lung, Heart, Trachea
    if mask_np.shape[0] != 3:
        raise ValueError(f"Expected mask with 3 channels, but got {mask_np.shape[0]} channels.")

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(config.device))  # Shape: (1, C, H, W)
        output = torch.sigmoid(output)
        pred_mask = (output > 0.5).float().cpu().numpy()[0]  # Shape: (C, H, W)

    # Define colors for each class: Lung (Red), Heart (Blue), Trachea (Green)
    colors = {
        0: (1, 0, 0),  # Red for Lung
        1: (0, 0, 1),  # Blue for Heart
        2: (0, 1, 0),  # Green for Trachea
    }

    # Create colored overlays for Ground Truth and Predictions
    gt_overlay = np.zeros_like(image_np)
    pred_overlay = np.zeros_like(image_np)

    for i in range(3):
        # Ground Truth Mask
        gt_class_mask = mask_np[i] > 0  # Binary mask for class i
        gt_overlay[gt_class_mask] = colors[i]

        # Predicted Mask
        pred_class_mask = pred_mask[i] > 0  # Binary mask for class i
        pred_overlay[pred_class_mask] = colors[i]

    # Set transparency factor
    alpha = 0.3

    # Overlay Ground Truth Masks on the original image
    gt_image = image_np.copy()
    gt_image = np.clip(gt_image + alpha * gt_overlay, 0, 1)

    # Overlay Predicted Masks on the original image
    pred_image = image_np.copy()
    pred_image = np.clip(pred_image + alpha * pred_overlay, 0, 1)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original Image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground Truth Mask Overlay
    axes[1].imshow(gt_image)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Predicted Mask Overlay
    axes[2].imshow(pred_image)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # Save the figure
    # plt.savefig(
    #     f"/workspace/code/lung-segmentation-tfm/resources/imgs/{model_name}_predictions_{index}.png",
    #     bbox_inches="tight",
    # )
    plt.show()


if not TRAIN:
    # Choose a few random samples from the validation dataset and visualize predictions
    for idx in range(5):
        # sample_idx = np.random.randint(0, len(test_loader.dataset))  # Randomly select an index
        sample_idx = idx  # For reproducibility, override with loop index if desired
        visualize_predictions(
            model_efficientUnet,
            test_loader.dataset,
            sample_idx,
            "model_efficientUnet",
        )  # Visualize the model's predictions

    # Choose a few random samples from the validation dataset and visualize predictions
    for idx in range(5):
        # sample_idx = np.random.randint(0, len(test_loader.dataset))  # Randomly select an index
        sample_idx = idx  # For reproducibility, override with loop index if desired
        visualize_predictions(
            model_unet,
            test_loader.dataset,
            sample_idx,
            "my_best_model",
        )  # Visualize the model's predictions

    # Choose a few random samples from the validation dataset and visualize predictions
    for idx in range(5):
        # sample_idx = np.random.randint(0, len(test_loader.dataset))  # Randomly select an index
        sample_idx = idx  # For reproducibility, override with loop index if desired
        visualize_predictions(
            model_unet_nb,
            test_loader.dataset,
            sample_idx,
            "model_notebook",
        )  # Visualize the model's predictions


# Final evaluation on the test set for the EfficientNet-based U-Net model
final_test_loss, final_test_dice, final_test_iou = validate_model_per_class(model_efficientUnet, test_loader, criterion)

# Print final test metrics
print(f"Final Test Loss EfficientUnet: {final_test_loss:.4f}")
print(f"Final Test Dice Coefficient EfficientUnet: {final_test_dice}")
print(f"Final Test Jaccard Index EfficientUnet: {final_test_iou}")
print("-" * 30)

# Final evaluation on the test set for the EfficientNet-based U-Net model
final_test_loss, final_test_dice, final_test_iou = validate_model_per_class(model_unet, test_loader, criterion)

# Print final test metrics
print(f"Final Test Loss Unet: {final_test_loss:.4f}")
print(f"Final Test Dice Coefficient Unet: {final_test_dice}")
print(f"Final Test Jaccard Index Unet: {final_test_iou}")
print("-" * 30)

# Final evaluation on the test set for the EfficientNet-based U-Net model
final_test_loss, final_test_dice, final_test_iou = validate_model_per_class(model_unet_nb, test_loader, criterion)

# Print final test metrics
print(f"Final Test Loss Unet Notebook: {final_test_loss:.4f}")
print(f"Final Test Dice Coefficient Unet Notebook: {final_test_dice}")
print(f"Final Test Jaccard Index Unet Notebook: {final_test_iou}")
print("-" * 30)
