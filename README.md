# Advanced Playlist Analysis and Reconstruction

This repository contains a Python-based pipeline designed for preprocessing music data, generating visual insights, clustering, and reconstructing playlists using machine learning techniques. It utilizes popular libraries like pandas, scikit-learn, and imbalanced-learn for data processing and modeling, alongside Optuna for hyperparameter optimization.

## Features

1. **Data Preprocessing**:
   - Handles missing values and inconsistent formats in features.
   - Generates new features such as `song_age` and `tempo_energy_ratio`.
   - Standardizes data and expands features using polynomial transformations.
   - Selects features with sufficient variance for modeling.

2. **Exploratory Data Visualization**:
   - Plots the distribution of songs by decade.
   - Generates a correlation heatmap for audio features.

3. **Clustering**:
   - Applies PCA for dimensionality reduction.
   - Determines optimal clusters using silhouette scores.
   - Performs K-means clustering to group songs into clusters.

4. **Model Training**:
   - Balances class distribution using ADASYN.
   - Optimizes RandomForestClassifier models using Optuna for:
     - Predicting user preferences.
     - Predicting release year categories.

5. **Evaluation**:
   - Generates classification reports for predictions.
   - Visualizes confusion matrices for model performance.

6. **Playlist Reconstruction**:
   - Predicts user and year labels for unlabeled data.
   - Reconstructs playlists based on predictions and saves them as CSV files.

Note that the pipeline expects a specific type of feature list. Contact me via mail for the dataset used.
