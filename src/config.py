# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Utility function to convert environment boolean strings
def str_to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

# General settings
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
VERBOSE = str_to_bool(os.getenv("VERBOSE", "True"))

# Dataset configuration
raw_categories = os.getenv("DATASET_CATEGORIES", "all").strip()
if raw_categories.lower() in {"none", "all"}:
    DATASET_CATEGORIES = None
else:
    DATASET_CATEGORIES = [cat.strip() for cat in raw_categories.split(",") if cat.strip()]

MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "1000"))

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_GPU = str_to_bool(os.getenv("USE_GPU", "False"))
GPU_DEVICE = int(os.getenv("GPU_DEVICE", "0"))

# Search configuration
TOP_K_RESULTS = int(os.getenv("NUM_SEARCH_RESULTS", "5"))
VECTOR_DIM = os.getenv("VECTOR_DIM")
VECTOR_DIM = int(VECTOR_DIM) if VECTOR_DIM and VECTOR_DIM.isdigit() else None
N_TREES = int(os.getenv("N_TREES", "100"))
SEARCH_K = int(os.getenv("SEARCH_K", "-1"))
ANNOY_INDEX_PATH = os.getenv("ANNOY_INDEX_PATH", "newsgroups_index.ann")
NUM_SEARCH_RESULTS = int(os.getenv("NUM_SEARCH_RESULTS", "5"))

# Augmented Generation configuration
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
GENERATION_MAX_LENGTH = int(os.getenv("GENERATION_MAX_LENGTH", "512"))
GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", "0.9"))
GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", "50"))
GENERATION_MAX_DOCS = int(os.getenv("GENERATION_MAX_DOCS", "5"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
