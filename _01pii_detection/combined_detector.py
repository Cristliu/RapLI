import re

try:
    from .flair_detector import FlairDetector
    from .presidio_detector import PresidioDetector
    from .chinese_pii_detector import ChinesePIIDetector
except ImportError:
    from flair_detector import FlairDetector
    from presidio_detector import PresidioDetector
    from chinese_pii_detector import ChinesePIIDetector


class CombinedPIIDetector:
    def __init__(self, dataset_name=None, use_transformer=False, transformer_config=None):
        """
        Initialize the combined PII detector
        
        Args:
            dataset_name: Name of the dataset
            use_transformer: Whether to use Transformer model (only valid for Chinese datasets)
            transformer_config: Transformer configuration parameters
        """
        self.dataset_name = dataset_name
        self.use_transformer = use_transformer
        
        # Default configuration parameters
        if transformer_config is None:
            transformer_config = {
                'use_regex': True,  # Enable regex detection
                'confidence_threshold': 0.7,  # Confidence threshold
                'min_entity_length': 2  # Minimum entity length
            }
        
        # Select detector language and strategy based on dataset type
        if dataset_name == 'spam_email':
            # Chinese dataset: Use integrated ChinesePIIDetector
            # Decide whether to enable Transformer function based on use_transformer parameter
            transformer_config['use_transformer'] = use_transformer
            
            # Create integrated Chinese PII detector
            self.chinese_detector = ChinesePIIDetector(**transformer_config)
            
            self.flair_detector = None  # Do not use Flair
            self.presidio_detector = None  # Do not use Presidio
            self.language = "zh"
            self.use_chinese_intelligent = True
            
            detector_type = "Integrated Transformer Chinese PII Detector" if use_transformer else "Rule-matching Chinese PII Detector"
            print(f"[OK] Using {detector_type} to process dataset: {dataset_name}")
        else:
            # English dataset: Use traditional combined detector
            self.chinese_detector = None
            self.flair_detector = FlairDetector(language="en")
            self.presidio_detector = PresidioDetector(language="en")
            self.language = "en"
            self.use_chinese_intelligent = False
            print(f"[OK] Using traditional combined detector to process English dataset: {dataset_name}")
            
        self.categories = [
            "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", 
            "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", 
            "QUANTITY", "TIME", "WORK_OF_ART", "IP_ADDRESS", "PHONE_NUMBER", 
            "URL", "EMAIL_ADDRESS", "DATE_TIME", "CREDIT_CARD", "BANK_ACCOUNT"
        ]

    def detect_pii(self, text):
        """Use different PII detection strategies based on dataset type"""
        
        if self.use_chinese_intelligent:
            # Chinese dataset: Use integrated Chinese detector
            return self.chinese_detector.detect_pii(text)
        
        else:
            # English dataset: Use traditional Flair+Presidio combination
            presidio_results = self.presidio_detector.detect_pii(text)
            flair_results = self.flair_detector.detect_pii(text)

            # Label source
            for res in presidio_results:
                res["source"] = "Presidio"
            for res in flair_results:
                res["source"] = "Flair"

            # Use dictionary to store (start, end) -> entity_info for deduplication and merging
            combined_map = {}
            for res in presidio_results:
                combined_map[(res["start"], res["end"])] = res
            for res in flair_results:
                if (res["start"], res["end"]) not in combined_map:
                    combined_map[(res["start"], res["end"])] = res

            combined_results = list(combined_map.values())
            return combined_results

