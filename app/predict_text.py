import os
import joblib
import pandas as pd

from app.config import Config
from app.feature_extraction import apply_pca_and_scale_model, extract_features


def predict_text(text: str):
    features =  pd.DataFrame(extract_features(text)).T

    features_scaled = apply_pca_and_scale_model(features)

    model = joblib.load(os.path.join(Config.MODELS_PATH, 'model.joblib'))

    prediction = model.predict(features_scaled)
    prediction_prob = model.predict_proba(features_scaled)

    print(f'Prediction for the input text: {prediction}')
    print(f'Prediction probabilities for the input text: {prediction_prob[0]}')
    
    return prediction, prediction_prob