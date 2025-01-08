import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
import optuna
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(file_path):
    print("Debug: Starting preprocessing...")
    df = pd.read_csv(file_path)

    # Feature engineering
    current_year = 2024
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(current_year).astype(int)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    df['song_age'] = current_year - df['release_year']
    df['tempo_energy_ratio'] = df['tempo'] / (df['energy'] + 1e-6)

    # Define base features
    base_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
        'song_age', 'tempo_energy_ratio'
    ]

    # Validate base features in DataFrame
    missing_features = [col for col in base_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in the dataset: {missing_features}")

    # Standardize features
    print("Debug: Standardizing features...")
    scaler = StandardScaler()
    df[base_features] = scaler.fit_transform(df[base_features])

    # Polynomial feature expansion
    print("Debug: Performing polynomial feature expansion...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[base_features])
    feature_names = poly.get_feature_names_out(base_features)

    # Select features with sufficient variance
    print("Debug: Applying variance threshold for feature selection...")
    selector = VarianceThreshold(threshold=0.1)
    selected_poly_features = selector.fit_transform(poly_features)
    selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]

    # Replace existing DataFrame with only the relevant features
    print("Debug: Adding selected polynomial features to DataFrame...")
    df_poly = pd.DataFrame(selected_poly_features, columns=selected_feature_names)
    df = pd.concat([df.reset_index(drop=True), df_poly.reset_index(drop=True)], axis=1)

    # Deduplicate columns if necessary
    print("Debug: Deduplicating columns...")
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"Debug: Preprocessing completed. Final DataFrame shape: {df.shape}")
    return df, selected_poly_features, selected_feature_names, scaler, base_features


def visualize_data(df):
    """
    Generate visualizations for exploratory analysis.
    """
    if 'release_year' in df.columns:
        df['decade'] = (df['release_year'] // 10) * 10

        plt.figure(figsize=(12, 6))
        df['decade'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Distribution of Songs by Decade')
        plt.xlabel('Decade')
        plt.ylabel('Number of Songs')
        plt.show()

    audio_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]
    existing_audio_features = [f for f in audio_features if f in df.columns]
    if existing_audio_features:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[existing_audio_features].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Audio Features')
        plt.show()


def perform_clustering(df, features, max_k=10):
    """
    Perform PCA for dimensionality reduction and K-means clustering.
    """
    print("Debug: Performing clustering with PCA...")
    X = np.nan_to_num(features)
    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X)

    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        silhouette_scores.append(silhouette_score(X_pca, labels))

    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"Optimal number of clusters: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)

    return df, optimal_k, pca


def train_optimized_models(X, y_user, y_year, selected_features):
    """
    Train models using ADASYN for class imbalance handling and Optuna for hyperparameter optimization.
    """
    print("\n[START] Model Training")

    # Convert X to DataFrame for column handling
    X = pd.DataFrame(X, columns=selected_features)

    # Handle class imbalance using ADASYN
    print("Debug: Balancing data using ADASYN...")
    adasyn = ADASYN(random_state=42)
    X_user, y_user = adasyn.fit_resample(X, y_user)
    X_year, y_year = adasyn.fit_resample(X, y_year)

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial, X_data, y_data):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        model = RandomForestClassifier(random_state=42, **params)
        scores = []
        for train_idx, val_idx in skf.split(X_data, y_data):
            model.fit(X_data.iloc[train_idx], y_data.iloc[train_idx])
            scores.append(model.score(X_data.iloc[val_idx], y_data.iloc[val_idx]))
        return np.mean(scores)

    # Optimize User Prediction Model
    print("Debug: Optimizing User Prediction Model...")
    study_user = optuna.create_study(direction='maximize')
    study_user.optimize(lambda trial: objective(trial, X_user, y_user), n_trials=20)
    user_model = RandomForestClassifier(random_state=42, **study_user.best_trial.params)
    user_model.fit(X_user, y_user)

    # Optimize Year Prediction Model
    print("Debug: Optimizing Year Prediction Model...")
    study_year = optuna.create_study(direction='maximize')
    study_year.optimize(lambda trial: objective(trial, X_year, y_year), n_trials=20)
    year_model = RandomForestClassifier(random_state=42, **study_year.best_trial.params)
    year_model.fit(X_year, y_year)

    print("\n[END] Model Training Completed")
    return user_model, year_model


def evaluate_models(user_model, year_model, X_test, y_test_user, y_test_year):
    """
    Evaluate models and display results.
    """
    print("Debug: Evaluating models...")
    user_preds = user_model.predict(X_test)
    print("User Prediction Report:")
    print(classification_report(y_test_user, user_preds))

    year_preds = year_model.predict(X_test)
    print("\nYear Prediction Report:")
    print(classification_report(y_test_year, year_preds))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_user, user_preds), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - User Prediction")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_year, year_preds), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Year Prediction")
    plt.show()


def reconstruct_playlists(df, user_model, year_model, scaler, selected_features):
    """
    Reconstruct playlists by predicting user and year labels using the trained models.
    Save playlists as CSV files.
    """
    print("\n[START] Playlist Reconstruction")

    # Include the 'cluster' feature in scaling
    print(f"Debug: Refitting scaler to include all selected features: {selected_features}")
    scaler.fit(df[selected_features])  # Refit the scaler to include all features used in training

    # Prepare unlabeled data for predictions
    X_unlabeled = df[selected_features].dropna()
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    # Make predictions for user and year
    df.loc[X_unlabeled.index, 'predicted_user'] = user_model.predict(X_unlabeled_scaled)
    df.loc[X_unlabeled.index, 'predicted_year'] = year_model.predict(X_unlabeled_scaled)

    # Finalize playlist reconstruction
    df['final_user'] = df['user'].fillna(df['predicted_user'])
    df['final_year'] = df['top_year'].fillna(df['predicted_year'])

    # Save playlists as CSVs
    playlists = df.groupby(['final_user', 'final_year'])
    for (user, year), playlist in playlists:
        sanitized_user = str(user).replace(' ', '_')
        filename = f"{sanitized_user}_{year}_playlist.csv"
        playlist.to_csv(filename, index=False)
        print(f"Saved: {filename}")

    print("\n[END] Playlist Reconstruction Completed")
    return df


# Main Script
# Ensure the scaler includes 'cluster' during refitting
print("Debug: Adding cluster to selected features and refitting scaler...")
df, poly_features, selected_features, scaler, base_features = preprocess_data("mixed_playlist.csv")
selected_features.append('cluster')  # Add cluster to selected features after clustering
visualize_data(df)

# Perform clustering and append cluster labels to features
df, optimal_k, pca = perform_clustering(df, poly_features, max_k=10)
X = np.hstack([poly_features, df['cluster'].values.reshape(-1, 1)])

y_user = df['user'].fillna('Unknown')
y_year = df['top_year'].fillna('Unknown')

# Train models and evaluate
user_model, year_model = train_optimized_models(X, y_user, y_year, selected_features)
evaluate_models(user_model, year_model, X, y_user, y_year)

# Reconstruct playlists
reconstructed_df = reconstruct_playlists(df, user_model, year_model, scaler, selected_features)
print("Playlist reconstruction completed.")
