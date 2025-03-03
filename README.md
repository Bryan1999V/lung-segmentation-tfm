# Lung Segmentation TFM

Development of lung segmentation and classification neural networks.

## Datasets

This section lists datasets that can be used for this application, covering various types of lung diseases. Each dataset
includes a short summary on implementation and importation.

### OSIC - Pulmonary Fibrosis Progression

This dataset provides baseline chest CT scans for each patient, along with associated clinical information. Each patient
has an image taken at `week = 0` and multiple follow-up visits over approximately 1-2 years. The image format is
DICOM (`.dcm`).

* **Training Set**: anonymized baseline CT scan data.
* **Test Set**: known baseline CT scan data.

This dataset is hosted on Kaggle. To download it, you need to install the Kaggle API. Follow the
[installation guide](https://www.kaggle.com/docs/api) to set it up. Once installed, use the following command to
download the dataset to your desired path:

```shell
# If you do not have the venv installed, run this command
poetry install --with dev

# Then
poetry run kaggle competitions download -c osic-pulmonary-fibrosis-progression -p {DATASET_PATH}
unzip {DATASET_ZIP_FILEPATH} -d {DATASET_PATH}
rm {DATASET_ZIP_FILEPATH}
```
