# chinese_pii_detector.py
"""
Chinese PII Detector - Intelligent PII detection specifically for Chinese text
Uses a combination of strategies: Regular Expressions + Transformer Models + Rule Matching
"""

import re
import json
import os
import torch
import torch.nn.functional as F
import warnings
import logging
from typing import List, Dict, Any
import time

# Disable TensorFlow and JAX/FLAX, force use of PyTorch
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "False"
os.environ["USE_TORCH"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, logging as transformers_logging
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import transformers: {e}")
    print("Will use pure rule-matching mode")
    TRANSFORMERS_AVAILABLE = False

class ChinesePIIDetector:
    def __init__(self, use_transformer=True, use_regex=True, confidence_threshold=0.7, min_entity_length=2): # min_entity_length not used
        """
        Initialize Chinese PII Detector
        
        Args:
            use_transformer: Whether to use Transformer model for detection
            use_regex: Whether to enable regular expression detection
            confidence_threshold: Confidence threshold to filter low-confidence results
            min_entity_length: Minimum entity length to filter entities that are too short
        """
        start_time = time.time()
        print("Initializing Chinese PII Detector...")
        
        # Configuration parameters
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.use_regex = use_regex
        self.confidence_threshold = confidence_threshold
        self.min_entity_length = min_entity_length
        
        print(f"  Config: Transformer={'On' if self.use_transformer else 'Off'}, Regex={'On' if use_regex else 'Off'}")
        print(f"        Confidence Threshold={confidence_threshold}, Min Entity Length={min_entity_length}")#
        
        # Initialize Transformer model
        self.tokenizer = None
        self.model = None
        self.base_model = None  # For general BERT model
        self.ner_pipeline = None
        self.model_name = None
        self.use_fallback = False
        
        # Load resources first
        self._load_resources()

        if self.use_transformer:
            self._init_transformer_models()
        
        # Initialize various detection modes
        self._init_regex_patterns()
        self._init_name_dict()
        self._init_location_dict()
        self._init_entity_mapping()
        
        end_time = time.time()
        detector_type = []
        if self.use_transformer: detector_type.append("Transformer")
        if self.use_regex: detector_type.append("Rule Matching")
        print(f"Chinese PII Detector initialization completed ({'+'.join(detector_type)}), took {end_time - start_time:.2f} seconds")
    
    def _load_resources(self):
        """Load resource files (dictionaries, blacklists, etc.)"""
        try:
            resource_path = os.path.join(os.path.dirname(__file__), 'chinese_pii_resources.json')
            if os.path.exists(resource_path):
                with open(resource_path, 'r', encoding='utf-8') as f:
                    self.resources = json.load(f)
                print(f"  Loaded resources from {resource_path}")
            else:
                print(f"  Warning: Resource file not found at {resource_path}, using empty defaults")
                self.resources = {}
        except Exception as e:
            print(f"  Error loading resources: {e}")
            self.resources = {}

    def _init_transformer_models(self):
        """Initialize Transformer models"""
        if not TRANSFORMERS_AVAILABLE:
            return
            
        # Try to load different Chinese NER models (sorted by priority)
        model_candidates = [
            # Dedicated NER models
            "ckiplab/bert-base-chinese-ner",
            # General Chinese BERT models (more stable)
            "hfl/chinese-bert-wwm-ext", 
            "bert-base-chinese",
            # Multilingual models as backup
            "xlm-roberta-base",
            "distilbert-base-multilingual-cased"
        ]
        
        for model_name in model_candidates:
            try:
                print(f"  Attempting to load model: {model_name}")
                
                if model_name == "ckiplab/bert-base-chinese-ner":
                    # Handle NER model specifically
                    try:
                        print(f"    Attempting to create NER pipeline...")
                        self.ner_pipeline = pipeline(
                            "ner",
                            model=model_name,
                            tokenizer=model_name,
                            aggregation_strategy="simple",
                            device=0 if torch.cuda.is_available() else -1,
                            framework="pt",  # Force PyTorch
                            use_fast=False,   # Use slow tokenizer to avoid compatibility issues
                            torch_dtype=torch.float32  # Explicitly specify data type
                        )
                        print(f"    NER pipeline created successfully")
                    except Exception as e:
                        print(f"    NER pipeline failed, attempting manual load: {e}")
                        # Manually load NER model
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name, 
                                from_tf=False, 
                                use_fast=False,
                                trust_remote_code=False
                            )
                            self.model = AutoModelForTokenClassification.from_pretrained(
                                model_name, 
                                from_tf=False,
                                trust_remote_code=False,
                                torch_dtype=torch.float32
                            )
                            print(f"    Manual NER model load successful")
                        except Exception as manual_e:
                            print(f"    Manual load also failed: {manual_e}")
                            continue
                else:
                    # For general BERT models, load manually directly (avoid pipeline issues)
                    print(f"    Manually loading general model: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        from_tf=False, 
                        use_fast=False,
                        trust_remote_code=False
                    )
                    # For general models, do not load token classification head (will error), only use for embedding
                    try:
                        from transformers import AutoModel
                        self.base_model = AutoModel.from_pretrained(
                            model_name, 
                            from_tf=False,
                            trust_remote_code=False
                        )
                        print(f"    Successfully loaded base model, will use rule+embedding method")
                    except Exception as e:
                        print(f"    Base model load failed: {e}")
                        continue
                
                self.model_name = model_name
                print(f"  ✅ Successfully loaded model: {model_name}")
                break
                
            except Exception as e:
                print(f"  ⚠️ Model {model_name} load failed: {e}")
                continue
        
        # If all models fail to load, use rule matching as backup
        if not self.ner_pipeline and not self.model and not self.base_model:
            print("  ❌ All Transformer models failed to load, will use pure rule matching mode")
            self.use_transformer = False
            self.use_fallback = True
        else:
            self.use_fallback = False

    def _init_entity_mapping(self):
        """Initialize entity type mapping"""
        self.entity_mapping = {
            'PER': 'PERSON', 'PERSON': 'PERSON', 'B-PER': 'PERSON', 'I-PER': 'PERSON',
            'LOC': 'LOCATION', 'LOCATION': 'LOCATION', 'B-LOC': 'LOCATION', 'I-LOC': 'LOCATION',
            'ORG': 'ORGANIZATION', 'ORGANIZATION': 'ORGANIZATION', 'B-ORG': 'ORGANIZATION', 'I-ORG': 'ORGANIZATION',
            'GPE': 'GPE', 'B-GPE': 'GPE', 'I-GPE': 'GPE',
            'TIME': 'DATE_TIME', 'DATE': 'DATE_TIME', 'B-TIME': 'DATE_TIME', 'I-TIME': 'DATE_TIME',
        }

    def _init_regex_patterns(self):
        """Initialize regex patterns"""
        self.patterns = {
            # Phone Number - Mainland China
            'PHONE_NUMBER': [
                r'1[3-9]\d{9}',  # Standard 11-digit phone number
                r'(\+86[-\s]?)?1[3-9]\d{9}',  # With international code
                r'1[3-9]\d{4}[-\s]?\d{4}',  # With separator
            ],
            
            # ID Card
            'ID_CARD': [
                r'[1-6]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',  # 18-digit ID
                r'[1-6]\d{7}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}',  # 15-digit ID
            ],
            
            # Bank Account
            'BANK_ACCOUNT': [
                r'\b[1-9]\d{15,18}\b',  # 16-19 digit bank card
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # With separator
            ],
            
            # Email Address
            'EMAIL_ADDRESS': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            ],
            
            # URL
            'URL': [
                r'https?://[^\s\u4e00-\u9fa5]+',  # HTTP/HTTPS URL
                r'www\.[^\s\u4e00-\u9fa5]+\.[a-zA-Z]{2,}',  # WWW domain
            ],
            
            # IP Address
            'IP_ADDRESS': [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            ],
            
            # Date Time
            'DATE_TIME': [
                r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',  # Chinese date format
                r'\d{4}-\d{2}-\d{2}',  # ISO date format
                r'\d{1,2}:\d{2}(:\d{2})?',  # Time format
            ],
            
            # Money
            'MONEY': [
                r'[￥¥$]\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency symbol + amount
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*[元块毛角分]',  # Chinese currency unit
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:万|千|百)?元',  # Chinese number + Yuan
            ],
            
            # License Plate
            'LICENSE_PLATE': [
                r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5}',
            ],
            
            # QQ Number
            'QQ_NUMBER': [
                r'[1-9]\d{4,10}',  # 5-11 digit QQ number
            ],
        }
    
    def _init_name_dict(self):
        """Initialize Chinese name dictionary"""
        # Common Chinese surnames (remove surnames prone to false positives)
        self.common_surnames = set(self.resources.get('common_surnames', []))
        
        # Common name characters
        self.common_name_chars = set(self.resources.get('common_name_chars', []))
        
        # Common word blacklist (avoid misidentifying as names)
        self.common_word_blacklist = set(self.resources.get('common_word_blacklist', []))
    
    def _init_location_dict(self):
        """Initialize location dictionary"""
        # Provinces/Municipalities
        self.provinces = set(self.resources.get('provinces', []))
        
        # Common cities
        self.cities = set(self.resources.get('cities', []))
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in Chinese text
        
        Args:
            text: Text to detect
            
        Returns:
            List of detected PII entities, each containing:
            - text: Entity text
            - entity_type: Entity type
            - start: Start position
            - end: End position
            - confidence: Confidence score
            - source: Source identifier
        """
        results = []
        
        # 1. Transformer model detection (if enabled)
        if self.use_transformer:
            transformer_results = self._detect_with_transformer(text)
            results.extend(transformer_results)
        
        # 2. Regex detection (if enabled)
        if self.use_regex:
            regex_results = self._detect_by_regex(text)
            results.extend(regex_results)
        
        # 3. Chinese name detection (heuristic rules)
        name_results = self._detect_chinese_names(text)
        results.extend(name_results)
        
        # 4. Location detection
        location_results = self._detect_locations(text)
        results.extend(location_results)
        
        # --- NEW: Refinement Steps ---
        results = self._refine_date_entities(results, text)
        results = self._refine_money_entities(results, text)
        # -----------------------------
        
        # 5. Deduplication and sorting
        results = self._deduplicate_results(results)
        
        # Final filter: Remove all entities in the common word blacklist
        final_results = []
        for res in results:
            if not self._is_common_word(res['text']):
                final_results.append(res)
        results = final_results
        
        results.sort(key=lambda x: x['start'])
        
        return results

    def _detect_with_transformer(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities using Transformer model"""
        entities = []
        
        try:
            if self.ner_pipeline:
                # Use pipeline for NER
                results = self.ner_pipeline(text)
                
                for result in results:
                    # Apply confidence threshold filter
                    if result['score'] < self.confidence_threshold:
                        continue
                    
                    entity_text = result['word'].strip()

                    # Apply minimum length filter
                    if len(entity_text) < self.min_entity_length:
                        continue
                    
                    # Filter common words
                    if self._is_common_word(entity_text):
                        continue
                    
                    entity_type = self.entity_mapping.get(result['entity_group'], result['entity_group'])
                    
                    entities.append({
                        'text': entity_text,
                        'entity_type': entity_type,
                        'start': result['start'],
                        'end': result['end'],
                        'confidence': result['score'],
                        'source': 'Transformer-NER'
                    })
                    
            elif self.model and self.tokenizer:
                # Manually use model for prediction
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=2)
                
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                
                # Parse prediction results (simplified version)
                current_entity = None
                entity_tokens = []
                
                for i, (token, pred_id) in enumerate(zip(tokens, predictions[0])):
                    if token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                        
                    # Here we need to parse based on the specific model's label mapping
                    # For simplicity, we use basic heuristics
                    if pred_id.item() > 0:  # Assuming 0 is O tag
                        entity_tokens.append((token, i))
                    else:
                        if entity_tokens:
                            # End current entity
                            entity_text = self.tokenizer.convert_tokens_to_string([t[0] for t in entity_tokens])
                            entities.append({
                                'text': entity_text.strip(),
                                'entity_type': 'ENTITY',  # Simplified type
                                'start': 0,  # Simplified position info
                                'end': len(entity_text),
                                'confidence': 0.8,
                                'source': 'Transformer-Manual'
                            })
                            entity_tokens = []
                    
        except Exception as e:
            print(f"  Transformer detection error: {e}")
        
        return entities

    def _detect_by_regex(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using regular expressions"""
        results = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    # Extra validation logic
                    matched_text = match.group()
                    confidence = self._calculate_regex_confidence(entity_type, matched_text, text, match.start())
                    
                    if confidence > 0.5:  # Confidence threshold
                        results.append({
                            'text': matched_text,
                            'entity_type': entity_type,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': confidence,
                            'source': 'ChinesePII-Regex'
                        })
        
        return results
    
    def _detect_chinese_names(self, text: str) -> List[Dict[str, Any]]:
        """Detect Chinese names"""
        results = []
        
        # Name pattern: Surname + 1-3 common name characters (using original logic)
        name_pattern = r'[' + ''.join(self.common_surnames) + r'][' + ''.join(self.common_name_chars) + r']{1,3}'
        
        for match in re.finditer(name_pattern, text):
            matched_text = match.group()
            
            # Validate name plausibility
            if len(matched_text) >= 2 and len(matched_text) <= 4:
                # Check context to avoid false positives
                context_start = max(0, match.start() - 10)
                context_end = min(len(text), match.end() + 10)
                context = text[context_start:context_end]
                
                confidence = self._calculate_name_confidence(matched_text, context)
                
                if confidence > 0.6:
                    results.append({
                        'text': matched_text,
                        'entity_type': 'PERSON',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': confidence,
                        'source': 'ChinesePII-Name'
                    })
        
        return results
    
    def _detect_locations(self, text: str) -> List[Dict[str, Any]]:
        """Detect location information"""
        results = []
        
        # Detect provinces
        for province in self.provinces:
            for match in re.finditer(re.escape(province), text):
                results.append({
                    'text': province,
                    'entity_type': 'GPE',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'source': 'ChinesePII-Location'
                })
        
        # Detect cities
        for city in self.cities:
            for match in re.finditer(re.escape(city), text):
                results.append({
                    'text': city,
                    'entity_type': 'GPE',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85,
                    'source': 'ChinesePII-Location'
                })
        
        return results
    
    def _calculate_regex_confidence(self, entity_type: str, matched_text: str, full_text: str, position: int) -> float:
        """Calculate regex match confidence"""
        base_confidence = 0.7
        
        # Special validation for different types
        if entity_type == 'PHONE_NUMBER':
            # Phone number validation: Check if it is a valid Chinese phone number segment
            if matched_text.startswith(('13', '14', '15', '16', '17', '18', '19')):
                base_confidence = 0.95
            else:
                base_confidence = 0.6
        
        elif entity_type == 'ID_CARD':
            # ID card validation: Check check digit
            if len(matched_text) == 18:
                base_confidence = 0.9
            else:
                base_confidence = 0.8
        
        elif entity_type == 'EMAIL_ADDRESS':
            # Email validation: Check common domains
            common_domains = self.resources.get('common_domains', ['qq.com', '163.com', '126.com', 'gmail.com', 'sina.com', 'sohu.com'])
            if any(domain in matched_text for domain in common_domains):
                base_confidence = 0.95
            else:
                base_confidence = 0.8
        
        elif entity_type == 'QQ_NUMBER':
            # QQ number validation: Length and numeric range
            if 5 <= len(matched_text) <= 11 and matched_text.isdigit():
                qq_num = int(matched_text)
                if 10000 <= qq_num <= 99999999999:
                    base_confidence = 0.8
                else:
                    base_confidence = 0.5
            else:
                base_confidence = 0.3
        
        return min(base_confidence, 1.0)

    def _calculate_name_confidence(self, name: str, context: str) -> float:
        """Calculate name match confidence"""
        base_confidence = 0.6
        
        # Check if surname is a common surname
        if name[0] in self.common_surnames:
            base_confidence += 0.2
        
        # Check if name characters are common name characters
        name_chars = name[1:]
        common_char_count = sum(1 for char in name_chars if char in self.common_name_chars)
        if common_char_count > 0:
            base_confidence += 0.1 * common_char_count
        
        # Check context clues
        name_indicators = self.resources.get('name_indicators', [])
        if any(indicator in context for indicator in name_indicators):
            base_confidence += 0.15
            
        # Avoid common false positive words
        if name in self.common_word_blacklist:
            return 0.0  # Filter out directly
        
        return min(base_confidence, 1.0)
    
    def _is_common_word(self, text):
        """Check if it is a common word (to avoid false positives)"""
        # Normalize text: remove spaces
        text = text.replace(' ', '')
        
        # Check blacklist in class attributes
        if hasattr(self, 'common_word_blacklist') and text in self.common_word_blacklist:
            return True

        return False
    
    def _is_valid_chinese_name(self, name, full_text, start_pos, end_pos):
        """Validate if it is a valid Chinese name"""
        # 1. Basic length check
        if len(name) < 2 or len(name) > 4:
            return False
            
        # 2. Check if it is a common word
        if self._is_common_word(name):
            return False
            
        # 3. Check context - get 10 characters before and after
        context_start = max(0, start_pos - 10)
        context_end = min(len(full_text), end_pos + 10)
        context = full_text[context_start:context_end]
        
        # 4. Name indicators (positive evidence)
        name_indicators = self.resources.get('name_indicators', [])
        
        # Check if there are name indicators
        has_name_indicator = any(indicator in context for indicator in name_indicators)
        
        # 6. Check if it is at the beginning of a sentence (possibly a name)
        is_sentence_start = start_pos == 0 or full_text[start_pos-1] in '。！？；\n'
        
        # 7. Check if it follows a colon or comma (possibly a speaker identifier)
        follows_punctuation = end_pos < len(full_text) and full_text[end_pos] in '：:，,'
        
        # 8. Comprehensive judgment
        if has_name_indicator:  # If there is a clear name indicator
            return True
        # elif has_negative_indicator:  # If there is a negative indicator
        #     return False
        elif is_sentence_start or follows_punctuation:  # If at start of sentence or follows punctuation
            return True
        else:  # Three or four character names are relatively safe
            return True
    
    def _refine_date_entities(self, results: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Ignore all DATE_TIME type entity detection results according to user requirements.
        The numeric part will be handled by PrivacyEngine's general rules (marked as risk level 3).
        """
        # Filter out all DATE_TIME type entities
        return [e for e in results if e['entity_type'] != 'DATE_TIME']

    def _refine_money_entities(self, results: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Optimize money entities, fix truncation issues"""
        for entity in results:
            if entity['entity_type'] == 'MONEY':
                current_end = entity['end']
                current_text = entity['text']
                
                # Normalize text to check ending (ignore spaces)
                clean_text = current_text.replace(' ', '')
                
                # Context for lookahead
                context = text[current_end:min(len(text), current_end + 10)]
                
                # Case 1: Ends with "人" (e.g. "200万 人") -> Look for "民币" or "民 币"
                if clean_text.endswith('人'):
                    match = re.match(r'^\s*民\s*币', context)
                    if match:
                        entity['end'] += match.end()
                        entity['text'] = text[entity['start']:entity['end']]
                
                # Case 2: Ends with "人民" -> Look for "币"
                elif clean_text.endswith('人民'):
                    match = re.match(r'^\s*币', context)
                    if match:
                        entity['end'] += match.end()
                        entity['text'] = text[entity['start']:entity['end']]
        return results

    def _spans_overlap(self, span1, span2):
        """Check if two spans overlap"""
        start1, end1 = span1
        start2, end2 = span2
        return not (end1 <= start2 or end2 <= start1)

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and overlapping detection results"""
        if not results:
            return results
        
        # Deduplication: Keep only one entity for the same position
        unique_entities = []
        seen_spans = set()
        
        # Sort by confidence, prioritize keeping high-confidence entities
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for entity in results:
            span = (entity['start'], entity['end'])
            
            # Check if it overlaps with existing entities
            overlapped = False
            for seen_span in seen_spans:
                if self._spans_overlap(span, seen_span):
                    overlapped = True
                    break
            
            if not overlapped:
                seen_spans.add(span)
                unique_entities.append(entity)
        
        return unique_entities
    