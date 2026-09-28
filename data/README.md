# Dataset

This project uses the **UCI Phishing Websites dataset**.

- **Source:** UCI Machine Learning Repository — Phishing Websites Data Set
  https://archive.ics.uci.edu/dataset/327/phishing+websites
- **Observations:** 11,055
- **Predictive features:** 30 (each encoded as -1 / 0 / 1)
- **Target variable:** `Result` (1 = legitimate, -1 = phishing)
- **Class distribution:** 6,157 legitimate, 4,898 phishing
- **Missing values:** none identified in the analysed copy of the dataset
- **Identifier column:** an `id` column is present and is excluded from
  modelling (see `src/data_preprocessing.py`)

## Obtaining the dataset

The dataset is not redistributed in this repository. To reproduce the
analysis:

1. Download the dataset from the UCI Machine Learning Repository link above
   (or an equivalent verified mirror).
2. Save it as a CSV file named:

   ```
   data/uci-ml-phishing-dataset.csv
   ```

3. The notebook and scripts in this repository expect the file at that
   relative path, with the column layout described above (30 feature
   columns, an `id` column, and a `Result` target column).

## Attribution

This dataset was produced by its original authors (Mohammad, Thabtah and
McCluskey) and is hosted by the UCI Machine Learning Repository. It was not
created as part of this project; it is used here solely for research and
educational purposes. Please refer to the UCI Machine Learning Repository
page above for the dataset's citation and licensing terms before
redistributing it elsewhere.
