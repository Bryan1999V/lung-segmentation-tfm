"""."""

from torch import optim
from torch.utils.data import DataLoader

from lun_segmentation import data, train
from unet import model

TRAIN_PATH = "/root/data/train"
TRAIN_MASKS_PATH = "/root/data/train_masks"
TRAIN_SPLIT_SIZE = 0.8  # 80% training, 20% validation

MASKS_IMAGE_OPACITY = 0.4

N_EPOCHS = 5
BATCH_SIZE = 5
LEARNING_RATE = 1e-4


if __name__ == "__main__":
    full_dataset = data.CTLungDataset(TRAIN_PATH, TRAIN_MASKS_PATH)
    data.plot_images_and_masks_overlapped(data.get_dataloader(full_dataset, BATCH_SIZE), MASKS_IMAGE_OPACITY)

    unet_model = model.UNet(in_channels=1, n_classes=1)
    train.train(
        unet_model,
        optimizer=optim.Adam(unet_model.parameters(), lr=LEARNING_RATE),
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        dataset=full_dataset,
        train_split_size=TRAIN_SPLIT_SIZE,
    )
