# pii_detection/presidio_detector.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
# from presidio_anonymizer import AnonymizerEngine
import time
import logging
import subprocess
import sys

# Silence all warning messages
logging.basicConfig(level=logging.ERROR)


def ensure_spacy_model(model_name):
    try:
        import spacy
        spacy.load(model_name)
        print(f"OK: SpaCy model {model_name} already exists")
        return True
    except OSError:
        print(f"ERROR: SpaCy model {model_name} not installed")
        if model_name == "zh_core_web_trf":
            print(f"Please install Chinese Transformer model:")
            print(f"pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_trf-3.4.0/zh_core_web_trf-3.4.0-py3-none-any.whl")
        elif model_name == "zh_core_web_sm":
            print(f"Please manually install Chinese model: pip install zh_core_web_sm-3.4.0-py3-none-any.whl")
        else:
            print(f"Please install model: python -m spacy download {model_name}")
        return False


class CustomPhoneRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [
            Pattern(name="phone_pattern", regex=r"\b\d{3}-\d{3}-\d{3}\b", score=0.7)
        ]
        super().__init__(supported_entity="PHONE_NUMBER", patterns=patterns)

class PresidioDetector:
    def __init__(self, language="en"):
        start_time = time.time()
        self.language = language
        
        try:
            if language == "zh" or language == "chinese":
                print(f"Configuring Presidio for Chinese language support...")
                
                try:
                    print("Checking SpaCy Chinese model...")
                    zh_model_ok = ensure_spacy_model("zh_core_web_trf")
                    
                    if zh_model_ok:
                        en_model_ok = ensure_spacy_model("en_core_web_lg")
                        
                        try:
                            configuration = {
                                "nlp_engine_name": "spacy",
                                "models": [
                                    {"lang_code": "zh", "model_name": "zh_core_web_trf"},                                    {"lang_code": "en", "model_name": "en_core_web_lg"}
                                ]
                            }
                            
                            provider = NlpEngineProvider(nlp_configuration=configuration)
                            nlp_engine = provider.create_engine()
                            
                            self.analyzer = AnalyzerEngine(
                                nlp_engine=nlp_engine,
                                supported_languages=["zh", "en"]
                            )
                            print(f"OK: Presidio configured with Chinese NLP engine (SpaCy)")
                        except Exception as config_error:
                            print(f"WARNING: Bilingual configuration failed, falling back to default configuration: {config_error}")
                            raise Exception("Bilingual NLP configuration failed, using backup configuration")
                    else:
                        raise Exception("Chinese SpaCy model unavailable, using backup configuration")
                    
                except Exception as nlp_error:
                    print(f"WARNING: Chinese NLP engine configuration failed, using default configuration: {nlp_error}")
                    self.analyzer = AnalyzerEngine()
                    print(f"OK: Presidio configured with default engine")
                    
            else:
                print(f"Configuring Presidio for English language support...")
                self.analyzer = AnalyzerEngine()
                
        except Exception as e:
            print(f"WARNING: Presidio configuration error, using default configuration: {e}")
            self.analyzer = AnalyzerEngine()
        
        custom_phone_recognizer = CustomPhoneRecognizer()
        self.analyzer.registry.add_recognizer(custom_phone_recognizer)
        

        
        end_time = time.time()
        print(f"presidio model loaded in {end_time - start_time:.2f} seconds.")

    def detect_pii(self, text):
        try:
            if self.language in ['zh', 'chinese']:
                if hasattr(self.analyzer, 'supported_languages') and 'zh' in self.analyzer.supported_languages:
                    results = self.analyzer.analyze(text=text, language='zh')
                else:
                    results = self.analyzer.analyze(text=text, language='en')
            else:
                results = self.analyzer.analyze(text=text, language='en')
        except Exception as e:
            print(f"PII analysis exception, using English default mode: {e}")
            try:
                results = self.analyzer.analyze(text=text, language='en')
            except Exception as fallback_e:
                print(f"Backup analysis also failed: {fallback_e}")
                results = []
        
        unique_results = []

        seen_spans = set()  # Used to store (start, end) ranges of all entities

        for result in results:
            span = (result.start, result.end)
            # Check if the entity range overlaps with existing ranges
            if not any(self._is_overlap(span, existing_span) for existing_span in seen_spans):
                seen_spans.add(span)  # Add new range
                unique_results.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "text": text[result.start:result.end]
                })
        
        return unique_results
    
    def _is_overlap(self, span1, span2):
        """Check if two ranges overlap"""
        start1, end1 = span1
        start2, end2 = span2
        return not (end1 <= start2 or end2 <= start1)
    
