import torch
import torch.nn.functional as F
import numpy as np
from nltk.corpus import stopwords
import random
import re
import json
import os

class DPNoiseAdder:### No need to decode first for classification tasks, as there are no special symbols like G
    
    # Class-level cache to avoid reloading Excel file repeatedly
    _cached_chinese_chars = None
    _cache_loaded = False
    
    def __init__(self, model, tokenizer, args, is_chinese=None):
        self.model = model
        self.tokenizer = tokenizer
        # Print confirmation of current tokenizer
        # print(f"Using tokenizer: {type(tokenizer)}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_matrix = model.get_input_embeddings().weight.to(self.device)  # (vocab_size, embedding_dim)
        
        # Load resources
        self._load_resources()

        # Get special token IDs
        self.special_token_ids = set(tokenizer.all_special_ids)  # Convert to set for faster lookup
        self.special_tokens = set(tokenizer.all_special_tokens)  # Convert to set for faster lookup
        print(f"Special tokens: {self.special_tokens}")
        
        # Set stop words and vocabulary filtering strategy based on dataset type
        self.args = args
        
        # Determine is_chinese flag
        if is_chinese is not None:
            self.is_chinese = is_chinese
        elif hasattr(args, 'dataset') and args.dataset in ['lcsts', 'spam_email']:  # Chinese dataset
            self.is_chinese = True
        else:
            self.is_chinese = False
            
        if self.is_chinese:
            # Chinese datasets do not use English stop words
            self.stop_words = set()
            print("Chinese mode detected, using Chinese vocabulary filtering strategy")
        else:  # English dataset
            self.stop_words = set(stopwords.words('english'))
            print("Using English vocabulary filtering strategy")

        # Pre-filter vocabulary (optimized version)
        self.filtered_vocab = self.filter_vocab_fast()
        print(f"Filtered vocab size: {len(self.filtered_vocab)}")
        
        # Pre-build vocabulary cache to improve runtime performance
        self._build_vocab_cache()

    def _load_resources(self):
        """Load resource files"""
        try:
            resource_path = os.path.join(os.path.dirname(__file__), 'dp_noise_resources.json')
            if os.path.exists(resource_path):
                with open(resource_path, 'r', encoding='utf-8') as f:
                    self.resources = json.load(f)
                print(f"Loaded resources from {resource_path}")
            else:
                print(f"Warning: Resource file not found at {resource_path}, using empty defaults")
                self.resources = {}
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.resources = {}
    
    @classmethod
    def _load_chinese_chars_cache(cls):
        """Class method: Cache loading of Chinese characters to avoid repeated reading of Excel file"""
        if cls._cache_loaded and cls._cached_chinese_chars is not None:
            return cls._cached_chinese_chars
            
        print("Loading Chinese characters from 3500 common vocabulary (first load)...")
        
        try:
            import pandas as pd
            
            # Build Excel file path
            excel_file_path = os.path.join(os.path.dirname(__file__), '3500_common_words.xls')
            
            if not os.path.exists(excel_file_path):
                raise FileNotFoundError(f"Vocabulary file not found: {excel_file_path}")
            
            # Fast read Excel file (read only first column)
            df = pd.read_excel(excel_file_path, usecols=[0], nrows=4000)  # Limit rows to improve speed
            
            # Fast extraction of Chinese characters
            common_chars = set()
            first_col = df.iloc[:, 0]  # Get first column
            
            for value in first_col.dropna():  # Remove null values
                text = str(value).strip()
                if len(text) == 1 and '\u4e00' <= text <= '\u9fff':  # Fast Unicode check
                    common_chars.add(text)
            
            print(f"Excel fast load completed, extracted {len(common_chars)} Chinese characters")
            
        except Exception as e:
            print(f"Excel load failed, using preset common character set: {e}")
            # Preset high-frequency character set to ensure basic functionality
            try:
                resource_path = os.path.join(os.path.dirname(__file__), 'dp_noise_resources.json')
                if os.path.exists(resource_path):
                    with open(resource_path, 'r', encoding='utf-8') as f:
                        resources = json.load(f)
                    common_chars = set(resources.get('common_chars_fallback', []))
                else:
                    raise Exception("Resource file not found")
            except:
                print("Failed to load preset common characters, using empty set")
                common_chars = set()
        
        # Cache results
        cls._cached_chinese_chars = common_chars
        cls._cache_loaded = True
        return common_chars
    


    def filter_vocab_fast(self):
        """Fast vocabulary construction, optimized version"""
        
        # Use cached common character set
        if self.is_chinese:
            common_chinese_chars = self._load_chinese_chars_cache()
            # Fast validation of available characters
            unk_id = self.tokenizer.unk_token_id
            valid_chinese_chars = {
                char for char in common_chinese_chars 
                if self.tokenizer.convert_tokens_to_ids(char) != unk_id
            }
        else:
            valid_chinese_chars = set()
        
        filtered_vocab = []
        vocab_size = self.embedding_matrix.size(0)
        
        # Pre-compile regex to improve performance
        chinese_pattern = re.compile(r'^[\u4e00-\u9fff]$')
        english_pattern = re.compile(r'^[a-zA-Z]+$')
        special_pattern = re.compile(r'^\[.*\]$')
        
        # Batch processing to reduce function call overhead
        batch_size = 2000  # Increase batch size
        for start_idx in range(0, vocab_size, batch_size):
            end_idx = min(start_idx + batch_size, vocab_size)
            
            for idx in range(start_idx, end_idx):
                try:
                    # Get token directly, avoid repeated decode calls
                    token = self.tokenizer.decode([idx]).strip()
                    
                    # Fast skip empty tokens or special tokens
                    if not token or token in self.special_tokens or special_pattern.match(token):
                        continue
                    
                    token_len = len(token)
                    
                    # Vocabulary filtering strategy for Chinese datasets (optimized version)
                    if self.is_chinese:
                        # 1. Single common Chinese character (highest priority)
                        if token_len == 1 and chinese_pattern.match(token) and token in valid_chinese_chars:
                            filtered_vocab.append(idx)
                        # 2. Two-character high-frequency Chinese words (improve replacement naturalness)
                        elif token_len == 2 and re.match(r'^[\u4e00-\u9fff]{2}$', token):
                            # Common two-character words, such as "的话", "时间", "问题", etc.
                            filtered_vocab.append(idx)
                        # 3. Three-character common words (moderately included)
                        elif token_len == 3 and re.match(r'^[\u4e00-\u9fff]{3}$', token):
                            filtered_vocab.append(idx)
                        # 4. Four-character words (selectively included, avoid overly long words)
                        elif token_len == 4 and re.match(r'^[\u4e00-\u9fff]{4}$', token):
                            # More filtering conditions can be added to ensure vocabulary quality
                            filtered_vocab.append(idx)
                        # 5. Common English words (for mixed text, narrow scope to improve quality)
                        elif re.match(r'^[a-zA-Z]{2,6}$', token) and token.islower():
                            # Limit length to 2-6 characters, ensure it is a common word
                            filtered_vocab.append(idx)
                        # 6. Number tokens (keep original logic)
                        elif re.match(r'^\d{1,4}$', token):
                            filtered_vocab.append(idx)
                    
                    # English datasets use simple filtering logic from May
                    elif not self.is_chinese:
                        # Check if token contains non-ASCII characters
                        if all(ord(c) < 128 for c in token):
                            # Exclude tokens containing only special symbols
                            if not re.search(r'[a-zA-Z0-9]', token) or re.match(r'^\[.*\]$', token):
                                continue
                            filtered_vocab.append(idx)
                    
                except:
                    # Silently skip tokens that fail to decode
                    continue
        return filtered_vocab
    
    def _build_vocab_cache(self):
        """Pre-build vocabulary cache to improve runtime performance"""
        print("Building vocabulary cache...")
        
        chinese_ids = []
        english_ids = []
        
        # Batch decode and classify (GPU friendly)
        vocab_tokens = self.tokenizer.batch_decode(self.filtered_vocab, skip_special_tokens=False)

        # Print some vocab_tokens examples for debugging
        print(f"Vocabulary examples (first 20): {vocab_tokens[:20]}")
        
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        english_pattern = re.compile(r'[a-zA-Z]')
        
        for i, token in enumerate(vocab_tokens):
            token = token.strip()
            has_chinese = chinese_pattern.search(token)
            has_english = english_pattern.search(token)
            
            if has_chinese and not has_english:
                chinese_ids.append(self.filtered_vocab[i])
            elif has_english and not has_chinese:
                english_ids.append(self.filtered_vocab[i])
        
        # Pre-calculate embedding matrix (GPU optimized)
        if chinese_ids:
            self._chinese_vocab_ids = chinese_ids
            self._chinese_embeddings = self.embedding_matrix[chinese_ids].to(self.device)
        
        if english_ids:
            self._english_vocab_ids = english_ids 
            self._english_embeddings = self.embedding_matrix[english_ids].to(self.device)
        
        self._vocab_cache = True
        print(f"Vocabulary cache build completed - Chinese vocab: {len(chinese_ids)}, English vocab: {len(english_ids)}")
    
    def filter_vocab(self):
        """Build high-quality mixed language vocabulary, intelligently filter rare words"""
        # Redirect to fast version
        return self.filter_vocab_fast()
    

    def calculate_k(self, epsilon, is_high_risk):###### Ablation Experiment 3: K_base takes different values
        if self.args.ablation_3_1:
            K_base = 10
        elif self.args.ablation_3_2:
            K_base = 50
        elif self.args.ablation_3_3:
            K_base = 100
        else:
            # Optimize K value for Chinese datasets
            if self.is_chinese:
                # Chinese optimization: More conservative K value, improve replacement quality
                if epsilon >= 8.0:
                    K_base = 8  # High epsilon uses small K value, improve precision
                elif epsilon >= 5.0:
                    K_base = 10  
                elif epsilon >= 1.0:
                    K_base = 12
                else:
                    K_base = 15  # Low epsilon allows more randomness
            else:
                K_base = 20  # English keeps original value
        
        # English datasets use simple logic from May
        if not self.is_chinese:
            if is_high_risk:
                return K_base 
            return int(K_base + round(100 / (epsilon ** 1.2)))
        ######### English dataset returns here
        
        # Optimization logic for Chinese datasets
        if is_high_risk:
            # Ensure high-risk tokens have enough candidates
            return max(12, K_base)  # At least 12 candidates
        
        # Dynamically adjust based on epsilon: Maintain standard behavior of differential privacy
        if epsilon >= 8.0:
            # High epsilon: Use smaller K value, improve replacement accuracy
            return max(3, min(K_base, 10))
        elif epsilon >= 5.0:
            # Medium epsilon: Moderate K value
            return max(5, min(K_base + 2, 15))
        elif epsilon >= 1.0:
            # Low epsilon: Slightly larger K value, maintain privacy
            return max(8, min(K_base + 5, 20))
        else:
            # Very low epsilon: Larger K value, ensure enough randomness
            dynamic_k = int(K_base + min(15, round(20 / (epsilon ** 0.8))))
            return max(10, min(dynamic_k, 25))

    def replace_number(self, token):
        # Check if it is an integer or decimal
        if re.match(r'^\d+(\.\d+)?$', token):
            original_num = float(token)
            
            # Chinese datasets use intelligent perturbation, English datasets use completely random from May
            if self.is_chinese and hasattr(self.args, 'epsilon'):
                epsilon = self.args.epsilon
                if epsilon >= 8.0:
                    # Very high epsilon (weak privacy protection): Tiny perturbation (within ±5%)
                    variation_rate = 0.05
                elif epsilon >= 5.0:
                    # High epsilon (weaker privacy protection): Small perturbation (within ±15%)
                    variation_rate = 0.15
                elif epsilon >= 2.0:
                    # Medium epsilon (medium privacy protection): Medium perturbation (within ±30%)
                    variation_rate = 0.3
                elif epsilon >= 1.0:
                    # Low epsilon (stronger privacy protection): Large perturbation (within ±50%)
                    variation_rate = 0.5
                else:
                    # Very low epsilon (strong privacy protection): Huge perturbation (within ±80%)
                    variation_rate = 0.8
                
                # Calculate perturbation range
                max_variation = max(1, int(original_num * variation_rate))
                min_val = max(1, int(original_num - max_variation))
                max_val = int(original_num + max_variation)
                
                # Generate new number (ensure it is not equal to original number)
                new_num = original_num
                while new_num == original_num:
                    new_num = random.randint(min_val, max_val)
                
                # Keep original number's integer/decimal format
                if '.' in token:
                    return f"{new_num:.1f}"
                else:
                    return str(new_num)
            else:
                # English dataset: Generate a completely random number (May version)
                new_num = original_num
                while new_num == original_num:
                    new_num = random.randint(1, 1000)
                return str(new_num)  # No space added
                
        # Check if it is an ordinal (e.g., 1st, 2nd, 3rd, 4th, etc.)
        elif re.match(r'^\d+(st|nd|rd|th)$', token):
            # Generate a new random ordinal
            new_num = random.randint(1, 100)
            # Determine ordinal suffix
            if 10 <= new_num % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(new_num % 10, 'th')
            return f"{new_num}{suffix}"  # No space added
        else:
            return token

    def replace_currency_symbol(self, token):
        # Replace currency symbol
        currency_symbols = self.resources.get('currency_symbols', ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦'])
        pattern = r'[' + re.escape(''.join(currency_symbols)) + r']'
        replacement_symbol = random.choice([sym for sym in currency_symbols if sym not in token])
        new_token = re.sub(pattern, replacement_symbol, token)

        return new_token


    def get_replacement(self, token, is_high_risk, epsilon_effective):
        """
        Generate candidate replacement word list and corresponding selection probabilities for a given token
        """
        # Save original token for debugging
        original_token = token
        
        # Get token ID
        target_token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_token_id = self.tokenizer.unk_token_id
        
        # Decode token and preprocess
        decoded_token = self.tokenizer.decode([target_token_id])
        decoded_token_lower = re.sub(r'\s+', '', decoded_token.lower())

        # Fast exit condition (improve performance)
        # Handle numbers or ordinals
        if re.match(r'^\d+(\.\d+)?$', decoded_token_lower) or re.match(r'^\d+(st|nd|rd|th)$', decoded_token_lower):
            new_token = self.replace_number(decoded_token_lower)
            return [new_token], [1.0]

        # Handle currency symbols
        currency_symbols = self.resources.get('currency_symbols', ['$', '€', '£', '¥', '₹', '₩', '₽', '₫', '฿', '₴', '₦'])
        if decoded_token_lower in currency_symbols:
            new_token = self.replace_currency_symbol(decoded_token_lower)
            return [new_token], [1.0]

        # Exclude special tokens
        if decoded_token_lower in self.special_tokens:
            return [token], [1.0]

        # Pure punctuation/special characters are kept directly
        if not re.search(r'[\u4e00-\u9fff]|[a-zA-Z0-9]', decoded_token_lower):
            return [token], [1.0]
        
        # Determine token language type
        has_chinese = re.search(r'[\u4e00-\u9fff]', decoded_token_lower)
        has_english = re.search(r'[a-zA-Z]', decoded_token_lower)
        has_digits = re.search(r'[0-9]', decoded_token_lower)
        
        # Fast skip strategy
        if has_english and not has_chinese:
            # English datasets use simple logic from May
            if decoded_token_lower.lower() in self.stop_words or not decoded_token_lower.isalpha():
                return [token], [1.0]
        elif has_chinese and not has_english:
            chinese_punctuation = self.resources.get('chinese_punctuation', '，。！？；：、""''（）《》【】……—–')
            if len(decoded_token_lower) == 1 and decoded_token_lower in chinese_punctuation:
                return [token], [1.0]
        elif has_chinese and has_english:
            return [token], [1.0]
        elif has_digits and not has_chinese and not has_english:
            return [token], [1.0]

        # GPU optimization: Get embedding
        if target_token_id != unk_token_id:
            # Get embedding directly from GPU memory (no CPU-GPU transfer)
            token_embedding = self.embedding_matrix[target_token_id:target_token_id+1]
        else:
            if self.args.ablation_4:
                return [token], [1.0]
            else:
                # Batch process UNK token (reduce model calls)
                with torch.no_grad():
                    inputs = self.tokenizer(token, return_tensors='pt').to(self.device)
                    outputs = self.model(**inputs)
                    token_embedding = outputs.last_hidden_state[:, 1:2, :].squeeze(1)

        # Get similar words (GPU accelerated)
        sim_words, sim_scores_normalized = self.get_top_k_similar_words(
            token_embedding, epsilon_effective, is_high_risk, original_token
        )

        # Fast probability calculation
        if not self.args.ablation_5 and is_high_risk:
            sim_scores_normalized = sim_scores_normalized[::-1]

        # Vectorized exponential mechanism calculation (GPU accelerated)
        delta_u = 1
        exponents = np.exp((epsilon_effective * sim_scores_normalized) / (2 * delta_u))
        probabilities = exponents / np.sum(exponents)

        return sim_words, probabilities
    
    def get_batch_replacements(self, tokens, is_high_risk_list, epsilon_effective):
        """
        Batch process multiple tokens to improve GPU utilization
        """
        if len(tokens) == 1:
            return [self.get_replacement(tokens[0], is_high_risk_list[0], epsilon_effective)]
        
        # Batch get token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        unk_id = self.tokenizer.unk_token_id
        
        results = []
        valid_tokens = []
        valid_indices = []
        
        # Pre-filter and fast exit
        for i, (token, token_id, is_high_risk) in enumerate(zip(tokens, token_ids, is_high_risk_list)):
            # Apply fast exit strategy
            decoded = self.tokenizer.decode([token_id]).strip().lower()
            
            # Fast processing for numbers, currency, special symbols, etc.
            if (re.match(r'^\d+', decoded) or 
                decoded in self.special_tokens or 
                not re.search(r'[\u4e00-\u9fff]|[a-zA-Z0-9]', decoded)):
                results.append(self.get_replacement(token, is_high_risk, epsilon_effective))
            else:
                valid_tokens.append((i, token, token_id, is_high_risk))
                valid_indices.append(i)
        
        if not valid_tokens:
            return results
        
        # Batch process tokens requiring embedding calculation
        embeddings_list = []
        for _, token, token_id, _ in valid_tokens:
            if token_id != unk_id:
                embeddings_list.append(self.embedding_matrix[token_id:token_id+1])
            else:
                if self.args.ablation_4:
                    embeddings_list.append(None)  # Mark skip
                else:
                    with torch.no_grad():
                        inputs = self.tokenizer(token, return_tensors='pt').to(self.device)
                        outputs = self.model(**inputs)
                        embeddings_list.append(outputs.last_hidden_state[:, 1:2, :].squeeze(1))
        
        # Batch calculate similar words
        batch_results = []
        for j, (orig_idx, token, token_id, is_high_risk) in enumerate(valid_tokens):
            if embeddings_list[j] is None:
                batch_results.append(([token], [1.0]))  # Fix: Unify return tuple format
                continue
                
            sim_words, sim_scores = self.get_top_k_similar_words(
                embeddings_list[j], epsilon_effective, is_high_risk, token
            )
            
            if not self.args.ablation_5 and is_high_risk:
                sim_scores = sim_scores[::-1]
            
            delta_u = 1
            exponents = np.exp((epsilon_effective * sim_scores) / (2 * delta_u))
            probabilities = exponents / np.sum(exponents)
            
            batch_results.append((sim_words, probabilities))
        
        # Merge results (ensure no None values)
        final_results = [None] * len(tokens)
        
        # First fill in processed fast exit results
        result_idx = 0
        for i in range(len(tokens)):
            if i not in valid_indices:
                final_results[i] = results[result_idx]
                result_idx += 1
        
        # Then fill in batch processing results
        for i, result in enumerate(batch_results):
            if i < len(valid_indices):
                orig_idx = valid_indices[i]
                final_results[orig_idx] = result
        
        # Ensure all positions have valid results (defensive programming)
        for i in range(len(final_results)):
            if final_results[i] is None:
                # If there are still None values, use original token as default
                final_results[i] = ([tokens[i]], [1.0])
            
        return final_results
    
    def get_top_k_similar_words(self, token_embedding, epsilon, is_high_risk, original_token="<unknown>"):
        K = self.calculate_k(epsilon,is_high_risk)
        
        # # Defensive check: Ensure K is not None
        # if K is None:
        #     print(f"Warning: K is None for epsilon={epsilon}, is_high_risk={is_high_risk}, is_chinese={self.is_chinese}")
        #     K = 20  # Use default value
        
        # Pre-cache vocabulary classification (avoid repeated calculation)
        if not hasattr(self, '_vocab_cache'):
            self._build_vocab_cache()
        
        # Select appropriate vocabulary based on original token language type
        has_chinese = re.search(r'[\u4e00-\u9fff]', original_token)
        has_english = re.search(r'[a-zA-Z]', original_token)
        
        # English datasets use simple logic from May: Use filtered_vocab directly
        if not self.is_chinese:
            selected_vocab_ids = self.filtered_vocab
            selected_embeddings = self.embedding_matrix[selected_vocab_ids]
        # Chinese optimization: Use stricter filtering strategy for Chinese characters
        elif has_chinese and not has_english and hasattr(self, '_chinese_vocab_ids'):
            selected_vocab_ids = self._chinese_vocab_ids
            selected_embeddings = self._chinese_embeddings
        elif has_english and not has_chinese and hasattr(self, '_english_vocab_ids'):
            selected_vocab_ids = self._english_vocab_ids  
            selected_embeddings = self._english_embeddings
        else:
            selected_vocab_ids = self.filtered_vocab
            selected_embeddings = self.embedding_matrix[selected_vocab_ids]

        # GPU batch similarity calculation (efficient parallel)
        with torch.no_grad():  # Reduce memory usage
            if self.args.ablation_6:
                # Batch Euclidean distance calculation
                distances = torch.cdist(token_embedding, selected_embeddings, p=2).squeeze(0)
                top_values, similar_indices = torch.topk(-distances, k=min(K, len(selected_vocab_ids)))
            else:            
                # Batch cosine similarity calculation (GPU optimized)
                similarities = F.cosine_similarity(
                    token_embedding.expand(selected_embeddings.size(0), -1), 
                    selected_embeddings, 
                    dim=1
                )
                top_values, similar_indices = torch.topk(similarities, k=min(K, len(selected_vocab_ids)))
        
        # Fast convert to numpy (reduce CPU-GPU transfer)
        indices_array = similar_indices.cpu().numpy()
        scores_array = top_values.detach().cpu().numpy()
        
        # Chinese optimization: Post-processing to filter higher quality similar words
        if has_chinese and not has_english and self.is_chinese:
            filtered_indices = []
            filtered_scores = []
            
            for idx, score in zip(indices_array, scores_array):
                candidate_id = selected_vocab_ids[idx]
                candidate_token = self.tokenizer.decode([candidate_id])
                
                # # Chinese character similarity validation
                # if self._is_good_chinese_replacement(original_token, candidate_token, score):
                filtered_indices.append(idx)
                filtered_scores.append(score)
                    
                # # If filtered candidates are too few, keep original results
                # if len(filtered_indices) >= max(3, K//3):
                #     break
            
            if len(filtered_indices) >= 3:
                indices_array = np.array(filtered_indices)
                scores_array = np.array(filtered_scores)
        
        # Ensure array is 1D (avoid dimension issues)
        if indices_array.ndim > 1:
            indices_array = indices_array.flatten()
        if scores_array.ndim > 1:
            scores_array = scores_array.flatten()
        
        # Initialize result list (fix UnboundLocalError)
        similar_words = []
        similar_scores_list = []
        
        for i, idx in enumerate(indices_array):
                try:
                    # Ensure idx is integer index and map to correct vocabulary ID
                    selected_vocab_idx = selected_vocab_ids[int(idx)]
                    vocab_idx = int(selected_vocab_idx)
                    decoded_word = self.tokenizer.decode([vocab_idx]).strip()
                    
                    # Intelligent similar word filtering: Ensure replacement quality
                    if decoded_word and decoded_word.strip():
                        decoded_word = decoded_word.strip()
                        
                        # Avoid replacing with special symbols and meaningless characters
                        if decoded_word in self.special_tokens:
                            continue
                        
                        # Avoid replacing with overly strange character combinations
                        if re.match(r'^[^\u4e00-\u9fff\w\s]+$', decoded_word):
                            continue
                        
                        # For Chinese, ensure basic quality of replacement words (independent of epsilon adjustment)
                        if self.is_chinese and re.search(r'[\u4e00-\u9fff]', original_token):
                            # Unified quality standard, not distinguishing epsilon values
                            if len(decoded_word) == 1 and re.match(r'[\u4e00-\u9fff]', decoded_word):
                                # Single character replacement
                                similar_words.append(decoded_word)
                                similar_scores_list.append(scores_array[i])
                            elif len(decoded_word) <= 3 and re.match(r'^[\u4e00-\u9fff]+$', decoded_word):
                                # Multi-character Chinese word
                                similar_words.append(decoded_word)
                                similar_scores_list.append(scores_array[i])
                        else:
                            # English datasets use simple processing from May (no extra filtering)
                            similar_words.append(decoded_word)
                            similar_scores_list.append(scores_array[i])
                        
                except (IndexError, TypeError, ValueError):
                    # Silently skip errors, do not output any info
                    continue        # If no valid similar words found, return default value
        if not similar_words:
            # Return default value (remove detailed debugging to improve performance)
            if self.is_chinese:
                similar_words = self.resources.get('default_chinese_replacements', ["的", "是", "在", "了", "和"])
                similar_scores_list = self.resources.get('default_chinese_scores', [1.0, 0.9, 0.8, 0.7, 0.6])
            else:
                similar_words = self.resources.get('default_english_replacements', ["the", "and", "is", "to", "a"])
                similar_scores_list = self.resources.get('default_english_scores', [1.0, 0.9, 0.8, 0.7, 0.6])

        # Normalize similarity scores
        sim_scores = np.array(similar_scores_list)
        if len(sim_scores) > 1:
            u_max = np.max(sim_scores)
            u_min = np.min(sim_scores)
            if u_max - u_min > 0:
                if self.args.ablation_6:### Ablation Experiment 6: Negative sign for Euclidean distance, min distance is max similarity
                    sim_scores_normalized = - (sim_scores - u_min) / (u_max - u_min)
                else:
                    sim_scores_normalized = (sim_scores - u_min) / (u_max - u_min)
            else:
                sim_scores_normalized = np.ones(len(sim_scores)) / len(sim_scores)
        else:
            sim_scores_normalized = np.array([1.0])

        return similar_words, sim_scores_normalized

    def _is_good_chinese_replacement(self, original_token, candidate_token, similarity_score):
        """
        Validate quality of Chinese character replacement
        """
        # Basic check: Candidate word must be Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', candidate_token):
            return False
            
        # Similarity threshold
        if similarity_score < 0.3:
            return False
            
        # Length check: Replacement word length should not differ too much
        if abs(len(original_token) - len(candidate_token)) > 1:
            return False
            
        # Avoid replacing with special symbols or numbers
        if re.search(r'[0-9\[\](){}]', candidate_token):
            return False
            
        # If single Chinese character, check if it is a common character
        if len(original_token) == 1 and len(candidate_token) == 1:
            # Validate from cached common characters
            if hasattr(self, '_cached_chinese_chars') and self._cached_chinese_chars:
                return candidate_token in self._cached_chinese_chars
        
        return True

    def _log_gpu_memory(self, step_name):
        """GPU memory usage monitoring"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU Memory] {step_name}: Allocated={memory_allocated:.2f}GB, Cached={memory_cached:.2f}GB")

    def select_replacement(self, sim_words, probabilities):
        """
        Select a replacement word from candidates based on probabilities
        """
        # Check if there are valid candidate words
        if not sim_words or len(sim_words) == 0:
            print("Warning: No similar words available, returning default token")
            return "的"  # Return a common Chinese character as default
        
        # Check if probability array is valid
        if len(probabilities) != len(sim_words) or np.sum(probabilities) == 0:
            print("Warning: Invalid probabilities, using uniform distribution")
            probabilities = np.ones(len(sim_words)) / len(sim_words)
        
        try:
            new_token = np.random.choice(sim_words, p=probabilities)
            return new_token
        except Exception as e:
            print(f"Warning: Error in token selection: {e}, returning first candidate")
            return sim_words[0] if sim_words else "的"