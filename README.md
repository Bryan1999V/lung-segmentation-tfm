# Lung Segmentation

Development of lung segmentation and classification neural networks.

## Datasets

This section lists datasets that can be used for this application, covering various types of lung diseases. Each dataset
includes a short summary on implementation and importation.

### [OSIC - Pulmonary Fibrosis Progression](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/data)

This dataset provides baseline chest CT scans for each patient, along with associated clinical information. Each patient
has an image taken at `week = 0` and multiple follow-up visits over approximately 1-2 years. The image format is DICOM
(`.dcm`). Dataset features:

- `DICOM` files (`.dcm`)
- No mask files are provided, so this dataset can only be used for **testing** the model.

This dataset is hosted on Kaggle. To download it, you need to install the Kaggle API. Follow the
[installation guide](https://www.kaggle.com/docs/api) to set it up. Once installed, use the following command to
download the dataset to your desired path:

```sh
# If you do not have the venv installed, run this command
poetry install --with dev

# Then
poetry run kaggle competitions download -c osic-pulmonary-fibrosis-progression -p {DATASET_PATH}
unzip {DATASET_ZIP_FILEPATH} -d {DATASET_PATH}
rm {DATASET_ZIP_FILEPATH}
```

Once the `zip` file is decompressed, the file tree structure is as follows:

```plain
/osic_dataset
|
├── test
│   ├── PATIENT_ID_1
│   │   ├── ...
│   │   └── N.dcm
│   ├── PATIENT_ID_2
│   │   ├── ...
│   │   └── N.dcm
│   ├── ...
│   └── PATIENT_ID_N
│       ├── ...
│       └── N.dcm
├── train
│   ├── ...
│   └── PATIENT_ID_N
│       ├── ...
│       └── N.dcm
├── sample_submission.csv
├── test.csv
└── train.csv
```

### [CT Lung & Heart & Trachea segmentation](https://www.kaggle.com/datasets/sandorkonya/ct-lung-heart-trachea-segmentation/data)

This dataset provides the segmentation masks for the _OSIC_ dataset to be used during the training process. There are
different classes mask: **lung**, **heart** and **trachea**. The image format is _`Nearly Raw Raster Data (.nrrd)`_.
Dataset features:

- `NRRD` files (`.nrrd`)
- Lung, heart and trachea masks for the _OSIC_ dataset
- The shape of each file data is `width x heigh x n_masks`

This dataset is hosted on kaggle too, so use the following commands to download it:

```sh
poetry install --with dev
poetry run kaggle datasets download --unzip -p {DATASET_PATH} sandorkonya/ct-lung-heart-trachea-segmentation
```

Once the `zip` file is decompressed, the file tree structure is as follows:

```plain
/lung_heart_trachea_segmentation
|
├── nrrd_heart
|   └── nrrd_heart
|       └── ...
├── nrrd_lung
|   └── nrrd_lung
|       ├── {PATIENT_ID_1}_lung.nrrd
|       ├── ...
|       └── {PATIENT_ID_N}_lung.nrrd
├── nrrd_noisy
|   └── nrrd_noisy
|       ├── {PATIENT_ID_1}_noisy.nrrd
|       ├── ...
|       └── {PATIENT_ID_N}_noisy.nrrd
└── nrrd_trachea
    └── nrrd_trachea
        └── ...
```

**_NOTE:_** _The masks located in the `nrrd_lung` folder was created by a closing morphological operation (grow 10mm,
shirnk 10mm) of the `nrrd_noisy` masks._
