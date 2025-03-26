"""."""

from lun_segmentation import data

BATCH_SIZE = 10

TRAIN_PATH = "/root/data/images"
TRAIN_MASKS_PATH = "/root/data/masks"
TRAIN_SPLIT_SIZE = 0.8


if __name__ == "__main__":
    application_dataset = data.CTLungDataset(TRAIN_PATH, TRAIN_MASKS_PATH)
    train_dataset, val_dataset = data.train_val_split(application_dataset, TRAIN_SPLIT_SIZE)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = data.get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data.get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    imgs_data, masks_data = next(iter(train_dataloader))
    print(f"Images data shape: {imgs_data.shape}")
    print(f"Masks data shape: {masks_data.shape}")

    data.plot_images_and_masks_overlapped(train_dataloader, mask_opacity=0.3)
