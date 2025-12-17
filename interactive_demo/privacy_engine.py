import sys
import os
import time
import torch
import numpy as np
import re
import json
import random
from transformers import AutoTokenizer, AutoModel

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01pii_detection.combined_detector import CombinedPIIDetector
from _02risk_assessment.risk_assessor import RiskAssessor
from _03dp_noise.noise_adder import DPNoiseAdder
from args import get_parser


class SimpleArgs:
    def __init__(self):
        # Default values based on args.py
        self.task = 'topic_classification'
        self.dataset = 'ag_news' # Default, can be changed
        self.model_name = "distilbert/distilbert-base-cased"
        self.epsilon = 1.0
        self.max_budget = 8.0
        self.EmbInver_K = 1
        self.MaskInfer_K = 1
        
        # Ablation flags
        self.ablation_1 = False
        self.ablation_2 = False
        self.ablation_3_1 = False
        self.ablation_3_2 = False
        self.ablation_3_3 = False
        self.ablation_4 = False
        self.ablation_5 = False
        self.ablation_6 = False
        
        self.new_ablation_1 = False
        self.new_ablation_2 = False
        
        self.use_transformer_pii = False
        self.pii_detection_mode = "intelligent"
        self.privacy_utility_balance = 0.5

class PrivacyEngine:
    def __init__(self, model_name="distilbert/distilbert-base-cased", dataset="ag_news"):
        print("Initializing PrivacyEngine with Integrated Resources...")
        self.args = SimpleArgs()
        self.args.dataset = dataset # Initial dataset
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load English Resources (DistilBERT)
        print("Loading English Model (DistilBERT)...")
        self.tokenizer_en = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
        self.model_en = AutoModel.from_pretrained("distilbert/distilbert-base-cased")
        self.model_en.to(self.device)
        self.noise_adder_en = DPNoiseAdder(self.model_en, self.tokenizer_en, self.args, is_chinese=False)
        
        # 2. Load Chinese Resources (Chinese-BERT-WWM)
        print("Loading Chinese Model (Chinese-BERT-WWM)...")
        self.tokenizer_cn = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.model_cn = AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.model_cn.to(self.device)
        self.noise_adder_cn = DPNoiseAdder(self.model_cn, self.tokenizer_cn, self.args, is_chinese=True)
        
        # 3. Initialize Integrated Detectors
        print("Initializing Integrated Detectors...")
        # English Detector (Traditional combination)
        self.detector_en = CombinedPIIDetector(dataset_name='ag_news', use_transformer=False)
        
        # Chinese Detector (Integrated with Transformer)
        self.detector_cn = CombinedPIIDetector(
            dataset_name='spam_email', 
            use_transformer=True,
            transformer_config={'use_regex': True, 'confidence_threshold': 0.7}
        )
        
        self.risk_assessor = RiskAssessor()
        
        # Load custom sensitive rules
        self.custom_sensitive_rules = self._load_custom_rules()
        
        # Set initial state
        self._set_active_resources(dataset)
        
        print("PrivacyEngine Initialized.")

    def _set_active_resources(self, dataset, text=None):
        """
        Selects the active model, tokenizer, detector, and noise adder based on dataset and text language.
        """
        self.args.dataset = dataset
        
        # Determine Language/Mode
        use_chinese = False
        
        if dataset == 'spam_email':
            use_chinese = True
        elif dataset == 'general':
            # Auto-detect for General Chat
            if text and re.search(r'[\u4e00-\u9fff]', text):
                use_chinese = True
            else:
                use_chinese = False
        else:
            # ag_news, samsum -> English
            use_chinese = False
            
        # Switch Resources
        if use_chinese:
            # print("Using Integrated Chinese Resources...")
            self.tokenizer = self.tokenizer_cn
            self.model = self.model_cn
            self.noise_adder = self.noise_adder_cn
            self.detector = self.detector_cn  # Now uses integrated detector
            self.args.use_transformer_pii = True
            self.args.pii_detection_mode = "transformer"
        else:
            # print("Using English Resources...")
            self.tokenizer = self.tokenizer_en
            self.model = self.model_en
            self.noise_adder = self.noise_adder_en
            self.detector = self.detector_en
            self.args.use_transformer_pii = False
            self.args.pii_detection_mode = "intelligent"

    def _load_custom_rules(self):
        rules = {}
        try:
            # 1. Try loading JSON rules
            json_path = os.path.join(os.path.dirname(__file__), 'custom_sensitive_rules.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                print(f"Loaded {len(rules)} custom sensitive rules from JSON.")
            
            # 2. Fallback/Merge: Try loading legacy txt (assign level 5)
            txt_path = os.path.join(os.path.dirname(__file__), 'custom_sensitive_words.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word and word not in rules:
                            rules[word] = 5
                print(f"Merged legacy custom words (Level 5). Total rules: {len(rules)}")
                
        except Exception as e:
            print(f"Error loading custom rules: {e}")
        return rules

    def calculate_privacy_budget(self, risk_level):
        if risk_level is None:
            return None
        return round(
            self.args.max_budget - (risk_level - 1) * ((self.args.max_budget - self.args.epsilon) / 4), 
            2
        )

    def analyze_text(self, text, dataset_name='ag_news'):
        """
        Analyzes text: cleans it, tokenizes it, detects PII, and maps PII to tokens.
        Returns data structure suitable for UI.
        """
        start_time = time.time()
        
        # Select resources based on dataset and text
        self._set_active_resources(dataset_name, text)
        
        # 1. Clean text (similar to main.py)
        if self.args.dataset in ['spam_email', 'general']:
            text = text.replace('—', '-')
            text = text.replace('…', '...')
            
        # --- 0. Initialize Risk Map & Apply Custom Rules FIRST ---
        char_risk_levels = {}
        custom_marked_indices = set()
        custom_safe_indices = set() # Track indices marked as explicitly safe
        
        # Apply custom sensitive rules
        for word, level in self.custom_sensitive_rules.items():
            # Treat None as 0 (Safe)
            is_safe = (level is None or level == 0)
            effective_level = 0 if is_safe else level
            
            # Case-insensitive search
            for match in re.finditer(re.escape(word), text, re.IGNORECASE):
                start, end = match.span()
                for i in range(start, end):
                    # Use max risk if multiple rules overlap
                    current_risk = char_risk_levels.get(i, 0)
                    char_risk_levels[i] = max(current_risk, effective_level)
                    
                    custom_marked_indices.add(i)
                    if is_safe:
                        custom_safe_indices.add(i)
            
        # 2. PII Detection
        pii_start = time.time()
        combined_results = self.detector.detect_pii(text)
        pii_time = time.time() - pii_start
        
        # 3. Risk Assessment (Initial)
        risk_start = time.time()
        assessed_results = self.risk_assessor.assess_risk(combined_results)
        
        # Merge Model Results into char_risk_levels
        for res in assessed_results:
            # Override for Transformer-NER ORGANIZATION
            if res.get('source') == 'Transformer-NER' and res.get('entity_type') == 'ORGANIZATION':
                res['risk_level'] = 3

            start = res['start']
            end = res['end']
            
            # Check for overlap with custom SAFE indices
            # If any part of the detected entity is explicitly marked as SAFE by user,
            # we discard the WHOLE entity to prevent partial false positives (e.g. "方面的" -> "的")
            entity_indices = set(range(start, end))
            if not entity_indices.isdisjoint(custom_safe_indices):
                continue
                
            for i in range(start, end):
                # Only apply if NOT marked by custom rules
                if i not in custom_marked_indices:
                    char_risk_levels[i] = res['risk_level']
        
        # --- Pre-tokenization Regex Detection (Fix for split tokens) ---
        # 1. Phone Numbers: 11 digits starting with 1
        # We check the original text to ensure we catch them even if tokenized into pieces
        for match in re.finditer(r'(?<!\d)1\d{10}(?!\d)', text):
            start, end = match.span()
            for i in range(start, end):
                if i not in custom_marked_indices:
                    char_risk_levels[i] = 5 # High Risk

        # 2. URLs: http/https/www
        for match in re.finditer(r'(https?://[a-zA-Z0-9./?=&_-]+|www\.[a-zA-Z0-9./?=&_-]+)', text):
            start, end = match.span()
            for i in range(start, end):
                if i not in custom_marked_indices:
                    char_risk_levels[i] = 3 # Mid Risk
        
        # # 3. Book Titles: 《...》
        # for match in re.finditer(r'《.*?》', text):
        #     start, end = match.span()
        #     for i in range(start, end):
        #         if i not in custom_marked_indices:
        #             char_risk_levels[i] = 3 # Mid Risk

        # 4. File Names: .xlsx, .pdf, .doc, .docx
        for match in re.finditer(r'[\w\u4e00-\u9fff]+\.(xlsx|pdf|docx?|pptx?|zip|rar)', text, re.IGNORECASE):
            start, end = match.span()
            for i in range(start, end):
                if i not in custom_marked_indices:
                    char_risk_levels[i] = 5 # High Risk
        
        # 5. Generic Numbers: Treat all numbers as at least Level 3 (Mid Risk)
        # This handles dates (e.g. 2025, 12, 8) and other numeric identifiers
        for match in re.finditer(r'\d+', text):
            start, end = match.span()
            for i in range(start, end):
                if i not in custom_marked_indices:
                    # Preserve existing higher risk (e.g. Phone numbers are 5)
                    current_risk = char_risk_levels.get(i, 0)
                    if current_risk < 3:
                        char_risk_levels[i] = 3
        # ---------------------------------------------------------------

        # 4. Tokenization
        encoded = self.tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        offsets = encoded['offset_mapping']
        
        # 5. Map to Tokens
        token_data = []
        for i, (token, (start_offset, end_offset)) in enumerate(zip(tokens, offsets)):
            token_text = text[start_offset:end_offset]
            
            # Check max risk for this token
            token_level = None
            for j in range(start_offset, end_offset):
                if j in char_risk_levels:
                    if token_level is None or char_risk_levels[j] > token_level:
                        token_level = char_risk_levels[j]
            
            # Default intelligent risk assignment (simplified from main.py)
            if token_level is None:
                # Skip whitespace/empty tokens
                if not token_text.strip():
                    token_level = 0
                else:
                    privacy_balance = getattr(self.args, 'privacy_utility_balance', 0.5)
                    perturbation_prob = 0.5 + (privacy_balance - 0.5)

                # For Chinese
                if self.args.dataset in ['spam_email', 'general']:
                    # Optimized chinese risk assignment strategy: reduce excessive perturbation
                    
                    # 1. Pure punctuation and special characters - No perturbation
                    if re.match(r'^[，。！？；：、""''（）《》【】…—–\s\$\+\-\*\/\=\<\>\[\]]+$', token_text):
                        token_level = None
                    
                    # 2. URL and email special formats - Medium risk
                    elif re.search(r'(https?://|www\.|@|\.(com|cn|org))', token_text):
                        # Special case: standalone @ symbol is safe
                        if token_text.strip() == '@':
                            token_level = None
                        else:
                            token_level = 3
                    
                    # 3. Pure numbers - High risk for phone numbers, others based on length
                    elif re.match(r'^\d+$', token_text):
                        # Phone number (11 digits, starts with 1)
                        if len(token_text) == 11 and token_text.startswith('1'):
                            token_level = 5
                        # Long numbers (possibly IDs, bank cards, etc.)
                        elif len(token_text) >= 6:
                            token_level = 5
                        else:
                            token_level = 3
                
                    
                    # English/Mixed - Default no perturbation
                    else:
                        token_level = None
                        
                else: 
                    token_level = None
                
            token_data.append({
                "index": i,
                "token": token,
                "text": token_text,
                "start": start_offset,
                "end": end_offset,
                "risk_level": token_level if token_level is not None else 0 # 0 means no risk/safe
            })
            
        # 6. Group tokens for UI (Reduce fatigue)
        visual_tokens = []
        last_end = 0
        
        for t in token_data:
            # Check if we should merge with previous
            # Condition: starts with ## (BERT) AND we have a previous group
            is_subword = t['token'].startswith('##')
            
            if is_subword and visual_tokens:
                # Merge
                prev = visual_tokens[-1]
                prev['token_indices'].append(t['index'])
                prev['text'] += t['text'] # Append the slice from original text
                prev['end'] = t['end']
                prev['risk_level'] = max(prev['risk_level'], t['risk_level'])
            else:
                # New visual word
                gap = text[last_end:t['start']]
                visual_tokens.append({
                    "text": t['text'],
                    "token_indices": [t['index']],
                    "risk_level": t['risk_level'],
                    "start": t['start'],
                    "end": t['end'],
                    "gap_before": gap
                })
            last_end = t['end']
            
        # Capture any trailing text
        trailing_text = text[last_end:]
        if trailing_text and visual_tokens:
             visual_tokens[-1]['gap_after'] = trailing_text
        elif trailing_text and not visual_tokens:
             # Case where no tokens were found but text exists (unlikely with BERT)
             pass

        risk_time = time.time() - risk_start
        total_time = time.time() - start_time
        
        return {
            "text": text,
            "tokens": token_data, # Keep raw tokens for reference if needed
            "visual_tokens": visual_tokens, # Optimized for UI
            "timings": {
                "pii_detection": pii_time,
                "risk_assessment": risk_time,
                "total_analysis": total_time
            }
        }

    def _detokenize_with_markers(self, tokens, risks):
        """
        Reconstructs text from tokens and risks, merging subwords and handling markers.
        """
        merged_tokens = []
        merged_risks = []
        
        for t, r in zip(tokens, risks):
            if t.startswith('##') and merged_tokens:
                merged_tokens[-1] += t[2:]
                # Take max risk for the merged token
                merged_risks[-1] = max(merged_risks[-1], r)
            else:
                merged_tokens.append(t)
                merged_risks.append(r)
        
        # Apply markers
        marked_parts = []
        for t, r in zip(merged_tokens, merged_risks):
            if r > 0:
                marked_parts.append(f"@@@{r}|{t}@@@")
            else:
                marked_parts.append(t)
        
        # Join
        text = " ".join(marked_parts)
        
        # Post-processing for punctuation and Chinese
        # 1. Remove spaces around Chinese characters (including those inside markers)
        # We need to be careful. 
        # Pattern: (Chinese/MarkerWithChinese) SPACE (Chinese/MarkerWithChinese)
        # It's easier to just use the regexes we already have, but adapted for markers.
        
        # Remove space between Chinese and Chinese
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        # Remove space between Chinese and Marker-starting-with-Chinese
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=@@@\d+\|[\u4e00-\u9fff])', '', text)
        # Remove space between Marker-ending-with-Chinese and Chinese
        text = re.sub(r'(?<=[\u4e00-\u9fff]@@@)\s+(?=[\u4e00-\u9fff])', '', text)
        # Remove space between Marker-ending-with-Chinese and Marker-starting-with-Chinese
        text = re.sub(r'(?<=[\u4e00-\u9fff]@@@)\s+(?=@@@\d+\|[\u4e00-\u9fff])', '', text)
        
        # 2. English Punctuation (Simple heuristic)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        text = text.replace(" ' ", "'").replace(" : ", ": ")
        
        # Fix URL spacing
        text = text.replace(" : / /", "://")
        text = text.replace(" . ", ".") # Be careful with this one, might affect sentences.
        # Better URL fix:
        text = re.sub(r'\s+:\s+//', '://', text)
        text = re.sub(r'(?<=http)s?\s+:\s+//', '://', text)
        text = re.sub(r'(?<=www)\s+\.', '.', text)
        text = re.sub(r'\s+\.(?=com|cn|org|net|edu|gov)', '.', text)
        
        return text

    def perturb_text(self, text, token_risk_map, manual_time=0.0, dataset_name='ag_news'):
        """
        Applies perturbation based on user-defined (or default) token risk levels.
        token_risk_map: dict {token_index: risk_level}
        manual_time: float, time spent by user on manual adjustment
        """
        start_time = time.time()
        
        # Select resources based on dataset and text
        self._set_active_resources(dataset_name, text)
        
        # Re-tokenize to ensure alignment (or assume text hasn't changed)
        # For safety, we re-tokenize
        if self.args.dataset in ['spam_email', 'general']:
            text = text.replace('—', '-')
            text = text.replace('…', '...')
            
        encoded = self.tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        
        # Calculate budgets
        token_privacy_budgets = []
        sensitive_tokens = []
        high_risk_tokens = []
        
        for i, token in enumerate(tokens):
            # Get risk level from map (default to 0 if not found)
            # Keys in JSON might be strings, so handle that
            risk_level = token_risk_map.get(str(i)) or token_risk_map.get(i) or 0
            risk_level = int(risk_level)
            
            if risk_level > 0:
                epsilon_i = self.calculate_privacy_budget(risk_level)
                token_privacy_budgets.append(epsilon_i)
                sensitive_tokens.append((token, epsilon_i))
                if risk_level >= 5: # Assuming 5 is high risk
                    high_risk_tokens.append(token)
            else:
                token_privacy_budgets.append(None)

        # Calculate epsilon_S
        if len(sensitive_tokens) == 0:
            epsilon_S = 100
        else:
            epsilon_S = sum([eps_i for eps_i in token_privacy_budgets if eps_i is not None]) / len(sensitive_tokens)
            
        # Generate replacements
        noise_start = time.time()
        new_tokens = []
        
        # Batch processing preparation
        tokens_to_process = []
        process_indices = []
        is_high_risk_list = []
        epsilon_effective_list = []
        
        for i, (token, epsilon_i) in enumerate(zip(tokens, token_privacy_budgets)):
            if epsilon_i is not None:
                tokens_to_process.append(token)
                process_indices.append(i)
                is_high_risk_list.append(token in high_risk_tokens)
                epsilon_effective_list.append(min(epsilon_S, epsilon_i))
        
        # Batch execution
        batch_results_map = {}
        if hasattr(self.noise_adder, 'get_batch_replacements') and len(tokens_to_process) > 0:
             # Use first epsilon for batch if they vary? 
             # get_batch_replacements takes a single epsilon_effective in the current code?
             # Let's check main.py: 
             # batch_results = noise_adder.get_batch_replacements(tokens_to_process, is_high_risk_list, epsilon_effective_list[0])
             # It seems it uses the first one. This might be a limitation in main.py or I should check noise_adder.
             # In noise_adder.py: get_batch_replacements(self, tokens, is_high_risk_list, epsilon_effective)
             # It takes a single scalar epsilon_effective.
             # So we should probably use the average or just pass one. main.py passes the first one.
             
             eff_eps = epsilon_effective_list[0] if epsilon_effective_list else epsilon_S
             batch_results = self.noise_adder.get_batch_replacements(
                tokens_to_process, is_high_risk_list, eff_eps
             )
             
             for idx, res in zip(process_indices, batch_results):
                 batch_results_map[idx] = res
        
        # Construct new tokens
        changes = []
        
        # Lists for reconstruction
        original_tokens_list = []
        original_risks_list = []
        perturbed_tokens_list = []
        perturbed_risks_list = []
        
        for i, (token, epsilon_i) in enumerate(zip(tokens, token_privacy_budgets)):
            # Determine risk level for this token
            risk_level = token_risk_map.get(str(i)) or token_risk_map.get(i) or 0
            risk_level = int(risk_level)
            
            original_tokens_list.append(token)
            original_risks_list.append(risk_level)

            if epsilon_i is not None:
                if i in batch_results_map and batch_results_map[i] is not None:
                    sim_words, probabilities = batch_results_map[i]
                    new_token = self.noise_adder.select_replacement(sim_words, probabilities)
                    
                    # # Force perturbation for high-risk numbers if unchanged
                    # if risk_level >= 5 and new_token == token and re.match(r'^\d+$', token):
                    #     # Generate random number of same length
                    #     new_token = "".join([str(random.randint(0, 9)) for _ in range(len(token))])

                    new_tokens.append(new_token)
                    
                    # Track changes
                    if new_token != token:
                        changes.append({
                            "index": i,
                            "original": token,
                            "perturbed": new_token
                        })
                    
                    perturbed_tokens_list.append(new_token)
                    perturbed_risks_list.append(risk_level) # Highlight perturbed with original risk
                else:
                    # Fallback
                    new_tokens.append(token)
                    perturbed_tokens_list.append(token)
                    perturbed_risks_list.append(0) # No change, no highlight in sanitized view
                    if new_token != token:
                         perturbed_risks_list[-1] = risk_level
                    else:
                         perturbed_risks_list[-1] = 0
            else:
                new_tokens.append(token)
                perturbed_tokens_list.append(token)
                perturbed_risks_list.append(0)
                
        noise_time = time.time() - noise_start
        
        # Reconstruct sentence
        perturbed_sentence = self.tokenizer.convert_tokens_to_string(new_tokens)
        
        # Use new detokenizer for marked text
        marked_original_sentence = self._detokenize_with_markers(original_tokens_list, original_risks_list)
        marked_sentence = self._detokenize_with_markers(perturbed_tokens_list, perturbed_risks_list)
        
        # Post-processing (Spaces) - simplified from main.py
        # Apply to perturbed_sentence (sent to LLM)
        if self.args.dataset in ['spam_email', 'general']:
             perturbed_sentence = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', perturbed_sentence)
             perturbed_sentence = re.sub(r'\s{2,}', ' ', perturbed_sentence)
        
        # Apply URL/Punctuation fixes to perturbed_sentence as well
        perturbed_sentence = perturbed_sentence.replace(" : / /", "://")
        perturbed_sentence = re.sub(r'\s+:\s+//', '://', perturbed_sentence)
        perturbed_sentence = re.sub(r'(?<=http)s?\s+:\s+//', '://', perturbed_sentence)
        perturbed_sentence = re.sub(r'(?<=www)\s+\.', '.', perturbed_sentence)
        perturbed_sentence = re.sub(r'\s+\.(?=com|cn|org|net|edu|gov)', '.', perturbed_sentence)

        total_time = time.time() - start_time
        
        # Group consecutive changes for better restoration context
        merged_changes = self._group_consecutive_changes(changes)

        return {
            "original_text": text,
            "perturbed_text": perturbed_sentence,
            "marked_perturbed_text": marked_sentence,
            "marked_original_text": marked_original_sentence,
            "changes": merged_changes,
            "epsilon_S": epsilon_S,
            "timings": {
                "noise_addition": noise_time,
                "total_perturbation": total_time,
                "manual_adjustment": manual_time
            }
        }

    def _group_consecutive_changes(self, changes):
        """
        Groups consecutive changes into a single block.
        This provides better context for restoration (e.g. "University of Louisville" -> "Academic are Indianapolis").
        """
        if not changes:
            return []
        
        # Sort by index
        changes.sort(key=lambda x: x['index'])
        
        merged = []
        current_group = [changes[0]]
        
        for i in range(1, len(changes)):
            prev = changes[i-1]
            curr = changes[i]
            
            # Check if consecutive
            if curr['index'] == prev['index'] + 1:
                current_group.append(curr)
            else:
                merged.append(self._process_change_group(current_group))
                current_group = [curr]
        
        # Process last group
        if current_group:
            merged.append(self._process_change_group(current_group))
            
        return merged

    def _process_change_group(self, group):
        # Extract tokens
        orig_tokens = [x['original'] for x in group]
        pert_tokens = [x['perturbed'] for x in group]
        
        # Clean tokens (remove ##) for regex building
        clean_orig_tokens = [t.replace('##', '') for t in orig_tokens]
        clean_pert_tokens = [t.replace('##', '') for t in pert_tokens]
        
        # Convert to string
        orig_text = self.tokenizer.convert_tokens_to_string(orig_tokens)
        pert_text = self.tokenizer.convert_tokens_to_string(pert_tokens)
        
        # Apply cleaning to perturbed text to match what is sent to LLM
        # 1. Chinese spacing
        if self.args.dataset in ['spam_email', 'general']:
             pert_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', pert_text)
        
        # 2. URL/Punctuation fixes (same as in perturb_text)
        pert_text = pert_text.replace(" : / /", "://")
        pert_text = re.sub(r'\s+:\s+//', '://', pert_text)
        pert_text = re.sub(r'(?<=http)s?\s+:\s+//', '://', pert_text)
        pert_text = re.sub(r'(?<=www)\s+\.', '.', pert_text)
        pert_text = re.sub(r'\s+\.(?=com|cn|org|net|edu|gov)', '.', pert_text)
        
        # 3. Remove spaces from subwords if they were joined with spaces by tokenizer
        # Clean up ## from original text (we want natural text)
        orig_text = orig_text.replace(' ##', '').replace('##', '')
        
        # For perturbed text, we usually want to keep ## if the LLM is expected to output them?
        # But the user said "Kattie" -> "Kit192".
        # If LLM outputs "Kit192", we want "Kit192".
        # If perturbed tokens are ["Kit", "##192"], string is "Kit192".
        # So we should also clean ## from perturbed text to match natural output.
        pert_text = pert_text.replace(' ##', '').replace('##', '')
        
        # Clean up spaces between digits (e.g. "1 2 3" -> "123")
        pert_text = re.sub(r'(?<=\d)\s+(?=\d)', '', pert_text)
        
        # Build flexible regex pattern
        regex_pattern = self._build_flexible_regex(pert_text)

        return {
            "index": group[0]['index'], # Start index
            "original": orig_text,
            "perturbed": pert_text,
            "original_tokens": clean_orig_tokens,
            "perturbed_tokens": clean_pert_tokens,
            "indices": [x['index'] for x in group],
            "regex_pattern": regex_pattern,
            "sub_changes": [
                {"original": x['original'].replace('##', ''), "perturbed": x['perturbed'].replace('##', '')}
                for x in group
            ]
        }

    def _build_flexible_regex(self, text):
        """
        Builds a flexible regex for the perturbed text.
        Splits text into atoms and allows flexible separators.
        """
        atoms = []
        # Split by whitespace first to get "words"
        raw_words = text.split()
        for word in raw_words:
            # Check if word has Chinese
            if re.search(r'[\u4e00-\u9fff]', word):
                # Split into chars, but keep consecutive English/Numbers together
                chunks = re.findall(r'[\u4e00-\u9fff]|[^\u4e00-\u9fff]+', word)
                atoms.extend(chunks)
            else:
                atoms.append(word)
        
        if not atoms:
            return re.escape(text)
            
        # Escape atoms
        escaped_atoms = [re.escape(a) for a in atoms]
        
        # Flexible separator: whitespace, dashes, commas, dots (optional)
        # We use non-capturing group (?:...) for efficiency if needed, but [] is fine
        separator = r"[\s\-_,.]*"
        
        regex = separator.join(escaped_atoms)
        
        # Enforce word boundaries for short alphabetic tokens to prevent false positives
        # e.g. prevent "d" matching inside "founded"
        # We check the original text (stripped) to decide if it's a short word
        clean_text = text.strip()
        if re.match(r'^[a-zA-Z]+$', clean_text) and len(clean_text) < 4:
            regex = r'\b' + regex + r'\b'
            
        return regex

    def _is_safe_to_replace_individually(self, token):
        """
        Determines if a token is distinctive enough to be replaced individually
        without strict context matching.
        """
        if not token: return False
        
        # 1. Length check
        if len(token) < 2: return False
        
        # 2. Common English Stopwords check (simplified)
        stopwords = {'the', 'and', 'are', 'for', 'not', 'was', 'this', 'that', 'with', 'from', 'have', 'one', 'two', 'but', 'all', 'in', 'on', 'at', 'to', 'of', 'is', 'it', 'be', 'as', 'or', 'by'}
        if token.lower() in stopwords: return False
        
        # 3. Length check for English
        if re.match(r'^[a-zA-Z]+$', token):
            # Skip short words unless they are capitalized (maybe?)
            # User said "Academy" (7) and "Shreveport" (10) are safe.
            # "one" (3) and "are" (3) are not.
            if len(token) < 4: return False 
            
        return True

    def restore_response(self, response, changes):
        """
        Restores the LLM response by replacing perturbed entities with original ones.
        changes: list of grouped changes {original, perturbed, original_tokens, perturbed_tokens}
        """
        start = time.time()
        restored = response
        
        # Sort changes by length of perturbed text (descending) to avoid partial matches
        sorted_changes = sorted(changes, key=lambda x: len(x['perturbed']), reverse=True)
        
        for change in sorted_changes:
            # 1. Prepare Original Text (Cleaned)
            o_text = change['original']
            # Remove spaces between Chinese characters in original text
            o_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', o_text)
            
            # 2. Use Flexible Regex
            pattern_str = change.get('regex_pattern')
            replaced = False
            
            if pattern_str:
                try:
                    # Use regex substitution
                    # We use re.IGNORECASE to be robust
                    # Check if it actually replaces anything
                    new_response, count = re.subn(pattern_str, o_text, restored, flags=re.IGNORECASE)
                    if count > 0:
                        restored = new_response
                        replaced = True
                except re.error:
                    pass
            
            if not replaced:
                # Fallback 1: Exact string match of the whole group
                if change['perturbed'] in restored:
                    restored = restored.replace(change['perturbed'], o_text)
                    replaced = True
            
            # Fallback 2: Individual distinctive tokens
            # If the group replacement failed, try to salvage distinctive parts
            if not replaced and 'sub_changes' in change:
                for sub in change['sub_changes']:
                    p_token = sub['perturbed']
                    o_token = sub['original']
                    
                    if self._is_safe_to_replace_individually(p_token):
                        # Try to replace p_token with o_token
                        # Use word boundaries for English
                        if re.match(r'^[a-zA-Z0-9]+$', p_token):
                             p_regex = r'\b' + re.escape(p_token) + r'\b'
                             try:
                                 restored = re.sub(p_regex, o_token, restored, flags=re.IGNORECASE)
                             except re.error:
                                 pass
                        else:
                             # Chinese or mixed, just replace
                             if p_token in restored:
                                 restored = restored.replace(p_token, o_token)
                
        return restored, time.time() - start

