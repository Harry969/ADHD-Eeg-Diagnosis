# Refining ADHD Diagnosis with EEG: The Impact of Preprocessing and Temporal Segmentation on Classification Accuracy

This project aims to improve the accuracy of ADHD diagnosis using EEG (Electroencephalography) data. The key focus is on refining preprocessing techniques and exploring the impact of temporal segmentation on the performance of various machine learning models.

## Project Overview

The project involves:
Preprocessing raw EEG data to make it suitable for machine learning.
Experimenting with different temporal segmentation strategies to enhance classification accuracy.
Evaluating the effectiveness of multiple machine learning algorithms (XGBoost, SVM, KNN, etc.) on the EEG dataset.
Exploring the impact of feature engineering and selection methods on the model's performance.

## Getting Started

To get started with the project, follow the steps below to set up the environment and run the notebooks.

### Requirements
Make sure to have Python 3.x installed along with the following libraries:
`numpy`
`pandas`
`mne`
`scikit-learn`
`xgboost`
`matplotlib`
`seaborn`

You can install the dependencies using:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas mne scikit-learn xgboost matplotlib seaborn
```
### Setup the Environment

1 Clone the repository:

```bash
git clone https://github.com/your-username/ADHD-Eeg-Diagnosis.git
cd ADHD-Eeg-Diagnosis
```
2 Set up a virtual environment (optional but recommended):

```bash
python -m venv eeg_env
source eeg_env/bin/activate  # On Windows use `eeg_env\Scripts\activate`
```

3 Install required packages:

```bash
pip install -r requirements.txt
```

4 Open Jupyter Notebooks:

```bash
jupyter notebook
```

## Running the Notebooks

You can start by running the notebook `1-preprocessing.ipynb` to preprocess the EEG data. Then, move on to model training and evaluation in the subsequent notebooks (e.g., `2-xgboost_1chan_all_feat.ipynb`, `3-svm_1chan_all_feat.ipynb`).

## Data

The dataset used in this project consists of EEG recordings, which are stored in `.pkl` files. These files contain the processed data and features that are used to train and evaluate machine learning models. Make sure to download and place them in the correct folder (`DATA/`).

## Data Processing

The `CollectData.py` script is used to collect the raw data and store it in the required format for further processing. The other scripts (`CreateChunks.py`, `CreateFeatures.py`) help in segmenting the data and extracting useful features for model training.

## Models and Evaluation

We experiment with several machine learning algorithms for classification, including:

- **XGBoost**
- **SVM**
- **KNN**

We also analyze how preprocessing and segmentation impact the overall performance, evaluating the models using metrics such as accuracy, recall, precision, and F1-score.

## Results

The results from the models can be found in the following `.pkl` files:

- `xgboost_1channel_df_results.pkl`
- `svm_1channel_df_results.pkl`
- `knn_1channel_df_results.pkl`

These files contain the prediction outcomes and evaluation metrics.

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch (`git checkout -b feature-branch`).
4. Make changes and commit them (`git commit -m "Add new feature"`).
5. Push to your fork and open a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

We thank the creators of the original dataset and the libraries used in this project. Special thanks to the contributors for their feedback and improvements.
