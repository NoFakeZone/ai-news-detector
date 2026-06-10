import os
import random
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import gc

# Upewnij się, że te importy pasują do Twoich plików
from dataset import NewsPopularityDataset
from load_dataset import load_dataset

# --- ZAKŁADAMY IMPORT 3 RÓŻNYCH KLAS MODELI ---
# Podmień je na swoje rzeczywiste klasy architektur!
from feature_bert import MultiModalBertModel, MultiModalBertModelMod, OnlyBert

# --- GLOBALNA KONFIGURACJA LOGOWANIA ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- WSPÓLNE PARAMETRY TRENINGOWE ---
BERT_MODEL_NAME = "allegro/herbert-base-cased"
NUM_CLASSES = 1 
BATCH_SIZE = 16
REAL_BATCH = 16
BATCH_ACCUMULATION = int(BATCH_SIZE / REAL_BATCH)
LEARNING_RATE = 2e-5
EPOCHS = 10
WARMUP_PROPORTION = 0.1 
DATA_PATH = r'C:\Users\PC\OneDrive\Pulpit\projekty\ai-news-generator'

# ==========================================
# --- LISTA KONFIGURACJI TRENINGOWYCH ---
# ==========================================
TRAINING_CONFIGURATIONS = [
    # ---------------------------------------------------------
    # WARIANT 1: OnlyBert (USE_STYLISTIC_FEATURES = False)
    # ---------------------------------------------------------
    # {
    #     "output_dir": "wb_newer/only_bert_gemini-2.5-flash",
    #     "model_class": OnlyBert, 
    #     "save_model": True,                  
    #     "test_data": "gemini-2.5-flash",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": False,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/only_bert_gemini-3-flash-preview",
    #     "model_class": OnlyBert, 
    #     "save_model": True,                  
    #     "test_data": "gemini-3-flash-preview",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": False,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/only_bert_gpt-oss-120b",
    #     "model_class": OnlyBert, 
    #     "save_model": True,                  
    #     "test_data": "gpt-oss-120b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": False,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/only_bert_llama-3.3-70b-instruct-fp8-fast",
    #     "model_class": OnlyBert, 
    #     "save_model": True,                  
    #     "test_data": "llama-3.3-70b-instruct-fp8-fast",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": False,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/only_bert_nemotron-3-120b-a12b",
    #     "model_class": OnlyBert, 
    #     "save_model": True,                  
    #     "test_data": "nemotron-3-120b-a12b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": False,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },

    # ---------------------------------------------------------
    # WARIANT 2: MultiModalBertModel (USE_STYLISTIC_FEATURES = True)
    # ---------------------------------------------------------
    # {
    #     "output_dir": "wb_newer/multimodal_gemini-2.5-flash",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-2.5-flash",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_gemini-3-flash-preview",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-3-flash-preview",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_gpt-oss-120b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gpt-oss-120b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_llama-3.3-70b-instruct-fp8-fast",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "llama-3.3-70b-instruct-fp8-fast",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_nemotron-3-120b-a12b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "nemotron-3-120b-a12b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },

    # ---------------------------------------------------------
    # WARIANT 3: MultiModalBertModelMod (USE_STYLISTIC_FEATURES = True)
    # ---------------------------------------------------------
    {
        "output_dir": "wb_newer/multimodal_mod_gemini-2.5-flash",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-2.5-flash",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_gemini-3-flash-preview",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-3-flash-preview",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_gpt-oss-120b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_llama-3.3-70b-instruct-fp8-fast",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "llama-3.3-70b-instruct-fp8-fast",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_nemotron-3-120b-a12b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "nemotron-3-120b-a12b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    # ---------------------------------------------------------
    # WARIANT 4: MultiModal + BASIC_POPULARITY_INDEX
    # ---------------------------------------------------------
    # {
    #     "output_dir": "wb_newer/multimodal_basic_pop_gemini-2.5-flash",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-2.5-flash",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": True,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_basic_pop_gemini-3-flash-preview",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-3-flash-preview",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": True,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_basic_pop_gpt-oss-120b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gpt-oss-120b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": True,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_basic_pop_llama-3.3-70b-instruct-fp8-fast",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "llama-3.3-70b-instruct-fp8-fast",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": True,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_basic_pop_nemotron-3-120b-a12b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "nemotron-3-120b-a12b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": True,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },

    # ---------------------------------------------------------
    # WARIANT 5: MultiModal + NKJP_POPULARITY_INDEX
    # ---------------------------------------------------------
    # {
    #     "output_dir": "wb_newer/multimodal_nkjp_pop_gemini-2.5-flash",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-2.5-flash",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": True,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_nkjp_pop_gemini-3-flash-preview",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-3-flash-preview",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": True,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_nkjp_pop_gpt-oss-120b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gpt-oss-120b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": True,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_nkjp_pop_llama-3.3-70b-instruct-fp8-fast",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "llama-3.3-70b-instruct-fp8-fast",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": True,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_nkjp_pop_nemotron-3-120b-a12b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "nemotron-3-120b-a12b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": False,
    #         "NKJP_POPULARITY_INDEX": True,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },

    # ---------------------------------------------------------
    # WARIANT 6: MultiModal + WIKIPEDIA_POPULARITY_INDEX (Normalne)
    # ---------------------------------------------------------
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_pop_gemini-2.5-flash",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-2.5-flash",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_pop_gemini-3-flash-preview",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-3-flash-preview",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_pop_gpt-oss-120b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gpt-oss-120b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_pop_llama-3.3-70b-instruct-fp8-fast",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "llama-3.3-70b-instruct-fp8-fast",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_pop_nemotron-3-120b-a12b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "nemotron-3-120b-a12b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
    #     }
    # },

    # ---------------------------------------------------------
    # WARIANT 7: MultiModal + WIKIPEDIA_POPULARITY_INDEX (UNNORMALIZED)
    # ---------------------------------------------------------
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_unnorm_pop_gemini-2.5-flash",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-2.5-flash",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_unnorm_pop_gemini-3-flash-preview",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gemini-3-flash-preview",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_unnorm_pop_gpt-oss-120b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "gpt-oss-120b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_unnorm_pop_llama-3.3-70b-instruct-fp8-fast",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "llama-3.3-70b-instruct-fp8-fast",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
    #     }
    # },
    # {
    #     "output_dir": "wb_newer/multimodal_wiki_unnorm_pop_nemotron-3-120b-a12b",
    #     "model_class": MultiModalBertModel, 
    #     "save_model": True,                  
    #     "test_data": "nemotron-3-120b-a12b",
    #     "flags": {
    #         "USE_STYLISTIC_FEATURES": True,
    #         "BASIC_POPULARITY_INDEX": False,
    #         "WIKIPEDIA_POPULARITY_INDEX": True,
    #         "NKJP_POPULARITY_INDEX": False,
    #         "NORMALIZE_NKJP": False,
    #         "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
    #     }
    # },
    # ---------------------------------------------------------
    # WARIANT 8: MultiModalMod + BASIC_POPULARITY_INDEX
    # ---------------------------------------------------------
    {
        "output_dir": "wb_newer/multimodal_mod_basic_pop_gemini-2.5-flash",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-2.5-flash",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_basic_pop_gemini-3-flash-preview",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-3-flash-preview",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_basic_pop_gpt-oss-120b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_basic_pop_llama-3.3-70b-instruct-fp8-fast",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "llama-3.3-70b-instruct-fp8-fast",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_basic_pop_nemotron-3-120b-a12b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "nemotron-3-120b-a12b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },

    # ---------------------------------------------------------
    # WARIANT 9: MultiModalMod + NKJP_POPULARITY_INDEX
    # ---------------------------------------------------------
    {
        "output_dir": "wb_newer/multimodal_mod_nkjp_pop_gemini-2.5-flash",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-2.5-flash",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": True,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_nkjp_pop_gemini-3-flash-preview",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-3-flash-preview",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": True,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_nkjp_pop_gpt-oss-120b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": True,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_nkjp_pop_llama-3.3-70b-instruct-fp8-fast",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "llama-3.3-70b-instruct-fp8-fast",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": True,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_nkjp_pop_nemotron-3-120b-a12b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "nemotron-3-120b-a12b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": True,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },

    # ---------------------------------------------------------
    # WARIANT 10: MultiModalMod + WIKIPEDIA_POPULARITY_INDEX (Normalne)
    # ---------------------------------------------------------
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_pop_gemini-2.5-flash",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-2.5-flash",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_pop_gemini-3-flash-preview",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-3-flash-preview",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_pop_gpt-oss-120b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_pop_llama-3.3-70b-instruct-fp8-fast",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "llama-3.3-70b-instruct-fp8-fast",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_pop_nemotron-3-120b-a12b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "nemotron-3-120b-a12b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict.json"
        }
    },

    # ---------------------------------------------------------
    # WARIANT 11: MultiModalMod + WIKIPEDIA_POPULARITY_INDEX (UNNORMALIZED)
    # ---------------------------------------------------------
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_unnorm_pop_gemini-2.5-flash",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-2.5-flash",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_unnorm_pop_gemini-3-flash-preview",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gemini-3-flash-preview",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_unnorm_pop_gpt-oss-120b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_unnorm_pop_llama-3.3-70b-instruct-fp8-fast",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "llama-3.3-70b-instruct-fp8-fast",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
        }
    },
    {
        "output_dir": "wb_newer/multimodal_mod_wiki_unnorm_pop_nemotron-3-120b-a12b",
        "model_class": MultiModalBertModelMod, 
        "save_model": True,                  
        "test_data": "nemotron-3-120b-a12b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": True,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False,
            "WIKIPEDIA_DICT_PATH": "wiki_popularity_dict_unnormalized.json"
        }
    }
]

def train_and_evaluate(config, device):
    output_dir = config["output_dir"]
    test_data = config["test_data"]
    flags = config["flags"]
    ModelClass = config["model_class"]  # Dynamiczne pobranie klasy
    save_model = config["save_model"]
    
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pt")
    checkpoint_path = os.path.join(output_dir, "latest_checkpoint.pt")
    log_path = os.path.join(output_dir, "training_log.txt")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"========== STARTING NEW TRAINING RUN ==========")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Model Architecture: {ModelClass.__name__}")
    logger.info(f"Saving Model To Disk: {save_model}")
    logger.info(f"Test Data: {test_data}")

    # --- 1. PRZYGOTOWANIE DANYCH ---
    data = load_dataset(
        test_data, DATA_PATH, flags["USE_STYLISTIC_FEATURES"], flags["BASIC_POPULARITY_INDEX"],
        flags["WIKIPEDIA_POPULARITY_INDEX"], flags["NKJP_POPULARITY_INDEX"], flags["NORMALIZE_NKJP"], flags["WIKIPEDIA_DICT_PATH"]
    )
    
    size_of_train = len(data[3])
    indices = random.sample(range(size_of_train), int(size_of_train * 0.1))

    test_text = list(data[0])
    test_labels = torch.tensor(data[1])
    test_features = torch.tensor(data[2])

    train_text = np.array(data[3])
    train_labels = torch.tensor(data[4])
    train_features = torch.tensor(data[5])

    val_text = train_text[indices]
    val_labels = train_labels[indices]
    val_features = train_features[indices]

    mask = torch.ones(len(train_text), dtype=torch.bool)
    mask[indices] = False

    train_text = list(train_text[mask])
    train_labels = train_labels[mask] 
    train_features = train_features[mask]
    val_text = list(val_text)

    logger.info(f"Train samples: {len(train_text)} | Val samples: {len(val_text)} | Test samples: {len(test_text)}")

    # --- 2. DYNAMICZNA INICJALIZACJA MODELU ---
    model = ModelClass( 
        bert_model_name=BERT_MODEL_NAME, 
        vector_input_size=train_features.shape[1] if train_features.numel() > 0 else 0, 
        num_classes=NUM_CLASSES
    )
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # =========================================================
    # --- 2.5 OBLICZANIE PARAMETRÓW DO STANDARD SCALING ---
    # =========================================================
    # Wyliczamy mean i std TYLKO na zbiorze treningowym (zapobiega data leakage)
    if train_features.numel() > 0 and flags["USE_STYLISTIC_FEATURES"]:
        feature_means = train_features.float().mean(dim=0)
        feature_stds = train_features.float().std(dim=0)
        # Zabezpieczenie na wypadek, gdyby std wynosiło dokładnie 0 (brak wariancji)
        feature_stds[feature_stds == 0] = 1e-8
    else:
        # Puste tensory, jeśli wektor cech jest pusty lub cechy są wyłączone
        feature_means = None
        feature_stds = None

    # --- 3. DATASETY I LOADERY ---
    # Przekazujemy te same statystyki (z train) do wszystkich 3 zbiorów
    train_dataset = NewsPopularityDataset(
        train_text, train_features, train_labels, BERT_MODEL_NAME, 
        use_features=flags["USE_STYLISTIC_FEATURES"], 
        feature_means=feature_means, feature_stds=feature_stds
    )
    val_dataset = NewsPopularityDataset(
        val_text, val_features, val_labels, BERT_MODEL_NAME, 
        use_features=flags["USE_STYLISTIC_FEATURES"], 
        feature_means=feature_means, feature_stds=feature_stds
    )
    
    train_loader = DataLoader(train_dataset, batch_size=REAL_BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=REAL_BATCH, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = math.ceil(len(train_loader) / BATCH_ACCUMULATION)
    total_training_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(total_training_steps * WARMUP_PROPORTION)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    best_val_loss = float('inf')
    best_model_state = None # Przechowujemy stan najlepszego w pamięci do testów

    # --- 4. GŁÓWNA PĘTLA TRENINGOWA ---
    for epoch in range(EPOCHS):
        logger.info(f"--- Starting Epoch {epoch+1}/{EPOCHS} ---")
        
        # TRENING
        model.train() 
        train_loss, train_correct, train_total = 0, 0, 0
        train_sum_target = 0.0
        optimizer.zero_grad() 
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            float_vectors = batch['float_vectors'].to(device)
            labels = batch['labels'].to(device).float().view(-1) 
            
            logits = model(input_ids, attention_mask, float_vectors).view(-1)
            loss = criterion(logits, labels)
            loss = loss / BATCH_ACCUMULATION 
            loss.backward()

            if ((batch_idx + 1) % BATCH_ACCUMULATION == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad() 

            train_loss += loss.item() * BATCH_ACCUMULATION 
            
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_sum_target += labels.sum().item()

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1} | Train Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item() * BATCH_ACCUMULATION:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # WALIDACJA
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                float_vectors = batch['float_vectors'].to(device)
                labels = batch['labels'].to(device).float().view(-1) 
                
                logits = model(input_ids, attention_mask, float_vectors).view(-1)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        logger.info(f">>> End of epoch {epoch+1}")
        logger.info(f"TRAIN -> Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2%}")
        logger.info(f"VAL   -> Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2%}")

        # Rejestrowanie najlepszego modelu
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Zapisujemy kopię stanu wag do pamięci RAM, żeby móc na niej przetestować
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if save_model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'feature_means': feature_means,  # ZAPISUJEMY NOWE STATYSTYKI
                    'feature_stds': feature_stds     # ZAPISUJEMY NOWE STATYSTYKI
                }
                torch.save(checkpoint, best_model_path)
                torch.save(checkpoint, checkpoint_path) # Nadpisz również najnowszy checkpoint
                logger.info(f"*** New best model saved to disk! (Val Loss: {best_val_loss:.4f}) ***")
            else:
                logger.info(f"*** New best model found (Val Loss: {best_val_loss:.4f}), but saving is disabled. ***")
        
        logger.info("-" * 50)

    # --- 5. TESTOWANIE NAJLEPSZEGO MODELU ---
    logger.info("--- Evaluating BEST model on TEST set ---")
    
    # Ładujemy do modelu najlepsze wagi zapamiętane z wariantu WALIDACJI (z ramu, a nie z dysku)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()

    # Tworzymy oddzielny dataset i loader testowy uzywając TYCH SAMYCH wartosci feature_means/stds ze zbioru treningowego
    test_dataset = NewsPopularityDataset(
        test_text, test_features, test_labels, BERT_MODEL_NAME, 
        use_features=flags["USE_STYLISTIC_FEATURES"], 
        feature_means=feature_means, feature_stds=feature_stds
    )
    test_loader = DataLoader(test_dataset, batch_size=REAL_BATCH, shuffle=False)

    test_loss, test_correct, test_total, test_sum_target = 0, 0, 0, 0
    test_tp, test_fp, test_fn = 0, 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            float_vectors = batch['float_vectors'].to(device)
            labels = batch['labels'].to(device).float().view(-1) 
            
            logits = model(input_ids, attention_mask, float_vectors).view(-1)
            loss = criterion(logits, labels)
            
            test_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            test_sum_target += labels.sum().item()

            test_tp += ((preds == 1) & (labels == 1)).sum().item()
            test_fp += ((preds == 1) & (labels == 0)).sum().item()
            test_fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_avg_target = test_sum_target / test_total

    precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
    recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f">>> FINAL TEST RESULTS FOR {output_dir} ({ModelClass.__name__}) <<<")
    logger.info(f"Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.2%} | Avg Target: {test_avg_target:.3f}")
    logger.info(f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1_score:.2%}")
    
    # --- 6. CZYSZCZENIE ZASOBÓW ---
    logger.removeHandler(file_handler)
    file_handler.close()

    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset, data
    del best_model_state
    
    if feature_means is not None:
        del feature_means, feature_stds
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Chosen DEVICE: {device}")
    logger.info(f"Found {len(TRAINING_CONFIGURATIONS)} configurations to run.\n")

    for idx, config in enumerate(TRAINING_CONFIGURATIONS):
        logger.info(f"=== INITIALIZING RUN {idx + 1}/{len(TRAINING_CONFIGURATIONS)} ===")
        try:
            train_and_evaluate(config, device)
        except Exception as e:
            logger.error(f"Error during training run '{config['output_dir']}': {e}")
            logger.info("Skipping to the next configuration...")

    logger.info("All training runs completed successfully!")

if __name__ == "__main__":
    main()