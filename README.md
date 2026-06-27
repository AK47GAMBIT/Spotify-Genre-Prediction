# 🎵 Spotify Track Genre Classification

A multi-class machine learning pipeline that predicts a Spotify track's **genre** from its audio features (danceability, energy, tempo, valence, etc.) using **XGBoost**.

---

## 📌 Project Overview

This project builds an end-to-end pipeline that:

1. Cleans and preprocesses raw Spotify track data
2. Explores relationships between audio features and genre (EDA)
3. Trains and compares multiple classification models
4. Tunes the best-performing model with `RandomizedSearchCV`
5. Engineers new features and performs feature selection based on importance
6. Saves the final trained model for inference on unseen data

---

## 🗂️ Repository Structure

```
.
├── model_training.ipynb     # Full training pipeline: EDA, model comparison, tuning, feature engineering
├── output.ipynb             # Inference notebook: loads saved artifacts and predicts on test.csv
├── train.csv                # Training data (with track_genre labels)
├── test.csv                 # Unlabeled data for inference
├── test_predictions.csv     # Output: test.csv + predicted_genre column
├── best_model.pkl           # Final tuned XGBoost classifier
├── label_encoder.pkl        # Fitted LabelEncoder for genre labels
├── selected_features.pkl    # List of feature columns the model expects, in order
└── README.md
```

---

## 🧹 Data Preprocessing

- Normalized the `explicit` column (mixed string/bool formats) into binary `0`/`1`
- Dropped non-predictive identifier columns: `Unnamed: 0`, `track_id`, `track_name`, `album_name`, `artists`
- Coerced `popularity` to numeric
- Dropped rows missing the target (`track_genre`)
- Filled missing numeric values **per genre group** (group-wise mean) rather than a global mean/median, to preserve genre-specific distributions
- Outliers were **kept** — they were judged to carry genuine signal rather than noise, and removing them would have shrunk an already noisy dataset

## 📊 Exploratory Data Analysis

- Boxplots of each audio feature vs. genre (full set, then narrowed to the top 10 genres by frequency for readability)
- Bar charts of average feature values per genre
- Count plot of `explicit` vs. genre
- Correlation heatmap across numeric features

## 🤖 Model Selection

Several classifiers were trained on the baseline feature set and compared on hold-out accuracy:

| Model | Accuracy |
|---|---|
| Logistic Regression | baseline |
| K-Nearest Neighbors | baseline |
| Random Forest | baseline |
| **XGBoost** | **best of the baseline models** |
| LightGBM | comparable |
| CatBoost | comparable |

XGBoost was selected as the strongest baseline and tuned further via `RandomizedSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` (3-fold CV, 10 candidate combinations).

## 🛠️ Feature Engineering

After initial tuning, additional features were engineered to help the model capture interactions between audio characteristics:

- **Interaction features:** `energy_loudness`, `tempo_energy`, `dance_energy`, `valence_energy`
- **Ratio features:** `energy_to_acoustic`, `speech_to_liveness`
- **Combined features:** `mood_score`, `intensity`
- **Skew correction:** log-transformed (`log1p`) `duration_ms` and `tempo`

**Feature selection** was then done using XGBoost's built-in feature importances, dropping any feature below an importance threshold of `0.01` (`mood_score` and `intensity` were dropped — their interaction effects were already captured by other retained features).

The final feature set used by the model is saved in `selected_features.pkl`.

## 📈 Final Model Performance

The final tuned XGBoost model classifies tracks into **114 genres** with:

- **Overall accuracy: ~60%**
- Macro avg F1-score: ~0.60

Given that this is a **114-class classification problem** on inherently noisy, overlapping audio-feature data (many genres have very similar sonic profiles, e.g. `rock` vs `alt-rock` vs `hard-rock`), 60% accuracy is a reasonable result. Performance varies notably by genre — distinctive genres like `grindcore` (0.90 F1), `comedy` (0.87 F1), `sleep` (0.86 F1), and `study` (0.85 F1) are classified very reliably, while acoustically similar/overlapping genres like `mpb` (0.39 F1) and `songwriter` (0.43 F1) are harder to separate.

> Accuracy could likely be improved further by restricting to a smaller set of top genres, merging closely related genres, or using more advanced ensembling — left as a future improvement.

---

## 🚀 Usage

### 1. Train from scratch
Run `model_training.ipynb` end-to-end on `train.csv`. This reproduces preprocessing, EDA, model comparison, hyperparameter tuning, feature engineering, and saves:
- `best_model.pkl`
- `label_encoder.pkl`
- `selected_features.pkl`

### 2. Run inference on new data
Run `output.ipynb`, which:
1. Loads `best_model.pkl` and `label_encoder.pkl`
2. Reads `test.csv` and re-creates the same engineered features used in training
3. Selects the columns listed in `selected_features.pkl`
4. Predicts genre labels and decodes them back from integers to genre names
5. Saves the result as `test_predictions.csv` (original columns + `predicted_genre`)

```python
import pickle
import pandas as pd
import numpy as np

# Load artifacts
best_model = pickle.load(open("best_model.pkl", "rb"))
le_final = pickle.load(open("label_encoder.pkl", "rb"))
selected_features = pickle.load(open("selected_features.pkl", "rb"))

# Load and engineer features on new data
df = pd.read_csv("test.csv")
df["energy_loudness"] = df["energy"] * df["loudness"]
df["tempo_energy"] = df["tempo"] * df["energy"]
df["dance_energy"] = df["danceability"] * df["energy"]
df["valence_energy"] = df["valence"] * df["energy"]
df["energy_to_acoustic"] = df["energy"] / (df["acousticness"] + 1e-5)
df["speech_to_liveness"] = df["speechiness"] / (df["liveness"] + 1e-5)
df["duration_ms"] = np.log1p(df["duration_ms"])
df["tempo"] = np.log1p(df["tempo"])

# Predict
X = df[selected_features]
preds = best_model.predict(X)
df["predicted_genre"] = le_final.inverse_transform(preds)
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

---

## 🔮 Possible Improvements

- Reduce the number of target classes (group rare/overlapping genres) to boost accuracy
- Try stacking/ensembling XGBoost with LightGBM and CatBoost
- Use audio embeddings or more granular audio features if available
- Address class imbalance more explicitly, if present, with class weights or resampling
