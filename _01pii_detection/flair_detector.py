# pii_detection/flair_detector.py
"""
Flair NER Detector
"""

from flair.data import Sentence
from flair.models import SequenceTagger
import time
import logging
import os
import shutil

# Simplified Flair detector, focusing on English NER

# Configure logging level
logging.basicConfig(level=logging.ERROR)

def clear_flair_cache_and_download(model_name):
    """Clear cache for a specific model and re-download"""
    try:
        # Get .flair/models path in user directory (Windows uses different path format)
        import platform
        if platform.system() == "Windows":
            # Windows system uses different user directory format
            home_dir = os.path.expanduser("~")
            flair_cache_dir = os.path.join(home_dir, ".flair", "models")
        else:
            flair_cache_dir = os.path.expanduser("~/.flair/models")
        
        # Ensure cache directory exists
        os.makedirs(flair_cache_dir, exist_ok=True)
        
        # Build model cache name and path
        model_cache_name = model_name.replace("flair/", "").replace("/", "-")
        model_cache_path = os.path.join(flair_cache_dir, model_cache_name)
        
        print(f"  Checking cache path: {flair_cache_dir}")
        print(f"  Model cache path: {model_cache_path}")
        
        # If cache file exists, delete it
        if os.path.exists(model_cache_path):
            print(f"  Clearing cache: {model_cache_path}")
            if os.path.isdir(model_cache_path):
                shutil.rmtree(model_cache_path)
            else:
                os.remove(model_cache_path)
        else:
            print(f"  Cache does not exist, preparing to download")
        
        # Try to re-download and load
        print(f"  Re-downloading model: {model_name}")
        
        # Set longer timeout to handle network issues
        return SequenceTagger.load(model_name)
        
    except Exception as e:
        print(f"  Failed to clear cache and re-download: {e}")
        return None

class FlairDetector:
    def __init__(self, language="en"):
        # Calculate loading time
        start_time = time.time()
        
        # Flair is mainly used for English NER, Chinese is handled by Presidio
        if language == "zh" or language == "chinese":
            print("  Chinese mode: Flair does not support Chinese, skipping Flair detection")
            self.tagger = None
        else:
            # English mode: Load English NER model
            try:
                print("  Attempting to load flair/ner-english-ontonotes-large model...")
                self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
                print("  [OK] Successfully loaded English NER model")
            except Exception as e1:
                print(f"  [WARNING] Large model loading failed: {e1}")
                try:
                    print("  Attempting to load flair/ner-english-fast model (backup)...")
                    self.tagger = SequenceTagger.load("flair/ner-english-fast")
                    print("  [OK] Successfully loaded English fast model")
                except Exception as e2:
                    print(f"  [Error] All Flair English models failed to load: {e2}")
                    self.tagger = None
            
        end_time = time.time()
        print(f"Flair detector initialization completed, took {end_time - start_time:.2f} seconds")

    def detect_pii(self, text):
        if self.tagger is None:
            return []
        
        try:
            sentence = Sentence(text)
            self.tagger.predict(sentence)
            
            # Get all detected entities
            all_entities = sentence.get_spans('ner')
            
            
            # Filter low confidence entities
            filtered_entities = [
                {
                    "text": entity.text,
                    "entity_type": entity.tag,
                    "start": entity.start_position,
                    "end": entity.end_position,
                    "confidence": entity.score
                }
                for entity in all_entities
                if entity.score > 0.5  # Only return entities with higher confidence
            ]
            
            if len(filtered_entities) != len(all_entities):
                print(f"    Retained {len(filtered_entities)} high confidence entities after filtering")
            
            return filtered_entities
            
        except Exception as e:
            print(f"    Error during Flair detection: {e}")
            return []