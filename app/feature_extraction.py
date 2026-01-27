
from typing import TypedDict
import joblib
import pandas as pd
import numpy as np
from app.config import Config
import spacy
import os
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lexical_diversity import lex_div as ld
from concurrent.futures import ThreadPoolExecutor


nlp = spacy.load('en_core_web_sm')
vectorizer = TfidfVectorizer(max_features=100)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


class SourceDataType(TypedDict):
    id: str
    label: str
    topic: str
    length_chars: int
    length_chars: int
    length_words: int
    quality_score: float
    sentiment: float
    source_detail: str
    timestamp: str
    plagiarism_score: float
    notes: str

def clean_text(text: str):
    # Your cleaning code here
    # Lowercase the text
    text = text.lower()

    # Remove leading and trailing whitespace
    text = text.strip()

    # Remove punctuation
    # text = text.replace(r'[^\w\s]', '', regex=True)

    # Long text truncation
    # text = text.slice(0, 500)
    
    return text

def lemmatize_text(text: str):

    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.text not in STOP_WORDS and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_)
            
    return ' '.join(tokens)

def get_readability_scores(text: str):
    # Your complexity scoring code here
    
    flesch_reading_ease_score = flesch_reading_ease(text)
    flesch_kincaid_grade_score = flesch_kincaid_grade(text)
    type_token_ratio = ld.ttr(text.split())
    
    return  pd.DataFrame([{
        'flesch_reading_ease': flesch_reading_ease_score,
        'flesch_kincaid_grade': flesch_kincaid_grade_score,
        'ttr': type_token_ratio
    }])

def extract_tf_idf_features(text: str):
    # TF-IDF Vectorization
    tf_idf_matrix = vectorizer.fit_transform([text])
    
    return pd.DataFrame(
                tf_idf_matrix.toarray(), 
                columns=vectorizer.get_feature_names_out()
            )


def get_sentence_embeddings(text: str):
    # Sentence Embedding using pre-trained models
    embeddings = sentence_transformer_model.encode([text])
    
    return pd.DataFrame(
        embeddings, 
        columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
    )


def get_llm_based_embeddings(text: str):
    # LLM-based Embedding using models like OpenAI's
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        
    embedding = outputs.last_hidden_state[0][0].cpu().numpy()
    
    return pd.DataFrame(
        [embedding], 
        columns=[f"llm_embedding_{i}" for i in range(len(embedding))]
    )


def extract_features(text):
    # Combined preprocessing function
    
    cleaned_text = clean_text(text)
    lemmatized_text = lemmatize_text(cleaned_text)
    
    with ThreadPoolExecutor() as executor:
        readability_scores = executor.submit(get_readability_scores, lemmatized_text).result()
        sentence_embeddings =  executor.submit(get_sentence_embeddings, text).result()
        llm_based_embeddings = executor.submit(get_llm_based_embeddings, text).result()
    
    features_df = pd.concat([
        readability_scores.reset_index(drop=True),
        sentence_embeddings.reset_index(drop=True),
        llm_based_embeddings.reset_index(drop=True)
    ], axis=1)
         
    return features_df.iloc[0]


def apply_pca_and_fit_scale(features_df) -> pd.DataFrame:
    # PCA Dimensionality Reduction
    
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)  # Retain 95% variance

    scaled_features_model = scaler.fit_transform(features_df)
    
    pca_features = pca.fit_transform(scaled_features_model)
    
    pca_features = pd.DataFrame(pca_features, columns=[f'pca_feature_{i}' for i in range(pca_features.shape[1])])
    
    joblib.dump(scaler, os.path.join(Config.MODELS_PATH, 'scaler_model.joblib'))
    joblib.dump(pca, os.path.join(Config.MODELS_PATH, 'pca_model.joblib'))
    
    return pca_features

def apply_pca_and_scale_model(features_df) -> pd.DataFrame:
    # PCA Dimensionality Reduction
    
    scaler_model = joblib.load(os.path.join(Config.MODELS_PATH, 'scaler_model.joblib'))
    pca_model = joblib.load(os.path.join(Config.MODELS_PATH, 'pca_model.joblib'))
    
    scaled_features = scaler_model.transform(features_df)
    
    pca_features = pca_model.transform(scaled_features)
    
    pca_features = pd.DataFrame(pca_features, columns=[f'pca_feature_{i}' for i in range(pca_features.shape[1])])
    
    return pca_features