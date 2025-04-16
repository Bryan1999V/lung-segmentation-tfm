"""."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim

from lun_segmentation import data, test, train
from unet import model

TRAIN_PATH = "/root/data/train"
TRAIN_MASKS_PATH = "/root/data/train_masks"
TEST_SPLIT_SIZE = 0.2
VAL_SPLIT_SIZE = 0.2
TEST_PATH = "/root/data/test/images"
TEST_MASK_PATH = "/root/data/test/masks"

MASKS_IMAGE_OPACITY = 0.4

TRAIN_MODEL = True
N_EPOCHS = 12
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
SEED = 42

torch.manual_seed(SEED)  # Set seed for torch operations on CPU
torch.cuda.manual_seed_all(SEED)  # Set seed for torch operations on GPU


if __name__ == "__main__":
    dataset = data.CTLungDataset(TRAIN_PATH, TRAIN_MASKS_PATH)
    train_set, test_set = train_test_split(
        dataset,
        test_size=TEST_SPLIT_SIZE,  # Proportion of data for validation
        random_state=SEED,  # Seed for reproducibility
    )
    test_loader = data.get_dataloader(test_set, batch_size=BATCH_SIZE, num_workers=2, train_mode=False)

    if TRAIN_MODEL:
        unet_model = model.UNet(in_channels=1, n_classes=1)
        history = train.train(
            unet_model,
            optimizer=optim.Adam(unet_model.parameters(), lr=LEARNING_RATE),
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            dataset=train_set,
            val_split_size=VAL_SPLIT_SIZE,
        )

        x_axis = np.arange(1, N_EPOCHS + 1, dtype=np.int32)

        # Plot Training and Validation Loss
        plt.figure()
        plt.plot(x_axis, history["loss"], label="Training Loss")
        plt.plot(x_axis, history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("/workspace/lung-segmentation-tfm/resources/imgs/training_loss_history.png")
        plt.close()

        # Plot Training and Validation Accuracy
        plt.figure()
        plt.plot(x_axis, history["accuracy"], label="Training Accuracy")
        plt.plot(x_axis, history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("/workspace/lung-segmentation-tfm/resources/imgs/training_accuracy_history.png")
        plt.close()

    else:
        unet_model = model.UNet(in_channels=1, n_classes=1)
        unet_model.load_state_dict(torch.load("/workspace/lung-segmentation-tfm/resources/models/my_best_model.pth"))
        unet_model.eval()

    num_samples_to_visualize = 5
    test.test_model(unet_model, test_set, num_samples_to_visualize)
