"""."""

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

from lung_segmentation import data, test, train, metrics
from unet import model

CSV_PATH = "/workspace/data/chest_ct_segmentation/train_filtered.csv"
IMAGES_PATH = "/workspace/data/chest_ct_segmentation/images/images"
MASKS_PATH = "/workspace/data/chest_ct_segmentation/masks/masks"
TEST_SPLIT_SIZE = 0.2
VAL_SPLIT_SIZE = 0.2

TRAIN_MODEL = False
N_EPOCHS = 8
N_WORKERS = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


if __name__ == "__main__":
    dataset = data.ChestCTDatasetCsv(IMAGES_PATH, MASKS_PATH, pd.read_csv(CSV_PATH))
    train_set, val_set, test_set = data.split_dataset(dataset, VAL_SPLIT_SIZE, TEST_SPLIT_SIZE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    criterion = metrics.BCEDiceLoss()
    unet_model = model.UNet(in_channels=3, n_classes=1)
    if TRAIN_MODEL:
        history = train.train(
            unet_model,
            N_EPOCHS,
            criterion,
            optim.Adam(unet_model.parameters(), lr=LEARNING_RATE),
            train_loader,
            val_loader,
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
        plt.show()

        # Plot Training and Validation Accuracy
        plt.figure()
        plt.plot(x_axis, history["accuracy"], label="Training Accuracy")
        plt.plot(x_axis, history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    else:
        unet_model.load_state_dict(torch.load("/root/models/best_unet_model_bryan.pth"))
        unet_model.eval()

    num_samples_to_visualize = 5
    test.test_model(unet_model, test_loader, criterion, num_samples_to_visualize)
