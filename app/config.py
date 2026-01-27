import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    MODELS_PATH =  os.path.join(BASE_DIR, "models")
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ai_human_text_classification.csv")
