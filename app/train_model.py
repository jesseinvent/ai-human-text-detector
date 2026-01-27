# Features extraction pipeline
import pandas as pd
from app.config import Config
from app.feature_extraction import apply_pca_and_fit_scale, extract_features
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    print("Loading dataset...")
        
    print(f"Dataset path: {Config.DATASET_PATH}")
    
    df = pd.read_csv(Config.DATASET_PATH)
    
    df.fillna('', inplace=True)
    
    print("Dataset loaded successfully.")
    
    print("Starting feature extraction...")
    
    features_df = df['text'].apply(extract_features)
    
    print("Feature extraction completed.")

    features_df.fillna(0, inplace=True)

    print("Applying PCA and scaling...")
    
    pca_scaled_features_df = apply_pca_and_fit_scale(features_df)
    
    print("PCA and scaling completed.")

    model = LogisticRegression()

    X = pca_scaled_features_df
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the model...")

    model.fit(X_train, y_train)
    
    print("Model training completed.")
    
    print("Saving the model...")

    joblib.dump(model, os.path.join(Config.MODELS_PATH, 'model.joblib'))
    
    print("Model saved successfully.")

    return True


if __name__ == "__main__":
    train_model()