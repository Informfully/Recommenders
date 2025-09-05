import random
import re
import numpy as np
import json
import pandas as pd
import gc
from typing import Dict, List, Any, Generator, Tuple

class NewsRecUtil:
    """
    Utility class for processing news recommendation data.
    Handles news title processing, user history management, and batch generation
    for neural news recommendation models.
    """
    
    def __init__(self, news_title=None, word_dict=None, impressionRating=None, 
                 user_history=None, history_size=50, title_size=30, max_cache_size=1000, batch_memory_limit=64):
        """
        Initialize NewsRecUtil with news data and configuration.
        
        Parameters:
        -----------
        news_title : dict
            Dictionary mapping news IDs to news titles
        word_dict : dict
            Dictionary mapping words to indices
        impressionRating : dict
            Dictionary containing positive and negative ratings for users
        user_history : dict
            Dictionary mapping user IDs to their historical interactions
        history_size : int
            Maximum number of historical articles to consider per user
        title_size : int
            Maximum number of words per news title
        max_cache_size : int
            Maximum number of items to keep in cache (default: 1000)
        batch_memory_limit : int
            Maximum batch size for memory efficiency (default: 64)
        """
        self.history_size = history_size  # Fixed typo from 'hisory_size'
        self.title_size = title_size
        self.impressionRating = impressionRating
        self.user_history = user_history
        self.news_title = news_title
        self.word_dict = word_dict
        self.click_title_all_users = {}

        # Caching mechanisms to improve performance
        self.user_history_cache = {}
        self.news_tokenization_cache = {}
        self._mappings_cached = False

        # Memory optimization settings
        self.max_cache_size = max_cache_size
        self.batch_memory_limit = batch_memory_limit  # Limit batch size for memory efficiency

        # Pre-allocated arrays for batch generation (will be initialized later)
        self._batch_arrays = None
        
    def newsample(self, news: List[int], ratio: int) -> List[int]:
        """
        Sample a specified number of items from news list.
        If length of news is less than ratio, pad with zeros.

        Parameters:
        -----------
        news : list
            Input news list with item indices
        ratio : int
            Number of samples to draw

        Returns:
        --------
        list
            Sampled news list, padded with zeros if necessary
        """
        if ratio > len(news):
            return news + [0] * (ratio - len(news))
        else:
            return random.sample(news, ratio)

    def load_data_from_file(self, train_set, npratio: int, batch_size: int) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Prepares and yields batches of training data from the given train_set.
        This is a memory-optimized generator that processes data in batches.

        Parameters:
        -----------
        train_set : object
            Training dataset containing user interactions in CSR matrix format
        npratio : int
            Negative sampling ratio (number of negative samples per positive sample)
        batch_size : int
            Size of each batch to yield

        Yields:
        -------
        dict
            Batch data containing:
            - user_index_batch: User indices
            - clicked_title_batch: Historical clicked news titles
            - candidate_title_batch: Candidate news titles (positive + negative)
            - labels: Binary labels (1 for positive, 0 for negative)
        """
        # Initialize news data if not already done
        if not hasattr(self, "news_title_index") or self.news_title_index is None:
            print("Initializing news data...")
            self.init_news(self.news_title)
        
        # Cache mappings to avoid repeated computation
        if not self._mappings_cached:
            self._cache_mappings(train_set)
        
        # Limit batch size for memory efficiency
        effective_batch_size = min(batch_size, self.batch_memory_limit)
        if effective_batch_size < batch_size:
            print(f"Reducing batch size from {batch_size} to {effective_batch_size} for memory efficiency")
        
        # Use optimized batch generator
        yield from self._optimized_batch_generator(train_set, npratio, effective_batch_size)

    def _cache_mappings(self, train_set) -> None:
        """
        Cache ID mappings to avoid repeated dictionary lookups.
        
        Parameters:
        -----------
        train_set : object
            Training dataset containing ID mappings
        """
        # Original item ID to Cornac item ID
        self.item_id2idx = train_set.iid_map
        # Cornac item ID to original item ID
        self.item_idx2id = {v: k for k, v in train_set.iid_map.items()}
        
        # Original user ID to Cornac user ID
        self.user_id2idx = train_set.uid_map
        # Cornac user ID to original user ID
        self.user_idx2id = {v: k for k, v in train_set.uid_map.items()}
        
        self._mappings_cached = True

    def _optimized_batch_generator(self, train_set, npratio: int, batch_size: int) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Memory-optimized batch generator using pre-allocated arrays.
        
        Parameters:
        -----------
        train_set : object
            Training dataset
        npratio : int
            Negative sampling ratio
        batch_size : int
            Batch size
            
        Yields:
        -------
        dict
            Batch data dictionary
        """
        if not hasattr(train_set, "uir_tuple"):
            raise ValueError("train_set does not contain the required 'uir_tuple' attribute.")

        # Get all unique user indices and shuffle them
        train_set_user_indices = list(set(train_set.uir_tuple[0]))
        np.random.shuffle(train_set_user_indices)

        # Pre-allocate numpy arrays for batch data (memory efficient)
        batch_labels = np.zeros((batch_size, npratio + 1), dtype=np.float32)
        batch_users = np.zeros((batch_size, 1), dtype=np.int32)
        batch_candidates = np.zeros((batch_size, npratio + 1, self.title_size), dtype=np.int64)
        batch_history = np.zeros((batch_size, self.history_size, self.title_size), dtype=np.int64)
        
        batch_idx = 0

        for user_idx in train_set_user_indices:
            try:
                # Get user's historical news titles (with caching)
                his_for_user = self._get_cached_user_history(user_idx)

                # Check if user has both positive and negative ratings
                if (user_idx in self.impressionRating["positive_rating"] and 
                    user_idx in self.impressionRating["negative_rating"]):
                    
                    train_pos_items = self.impressionRating["positive_rating"][user_idx]
                    train_neg_items = self.impressionRating["negative_rating"][user_idx]

                    if len(train_pos_items) > 0:
                        for p in train_pos_items:
                            # Create label: [1, 0, 0, ..., 0] for positive + negatives
                            batch_labels[batch_idx, 0] = 1.0  # Positive sample
                            batch_labels[batch_idx, 1:] = 0.0  # Negative samples
                            
                            # Set user index
                            batch_users[batch_idx, 0] = user_idx
                            
                            # Sample negative items
                            n = self.newsample(train_neg_items, npratio)
                            candidate_keys = [p] + n
                            
                            # Fill candidate titles directly into pre-allocated array
                            self._fill_candidate_titles(batch_candidates[batch_idx], candidate_keys)
                            
                            # Fill user history
                            batch_history[batch_idx] = his_for_user
                            
                            # Cache click history for this user
                            self.click_title_all_users[user_idx] = his_for_user
                            
                            batch_idx += 1

                            # Yield batch when it's full
                            if batch_idx >= batch_size:
                                yield {
                                    "user_index_batch": batch_users.copy(),
                                    "clicked_title_batch": batch_history.copy(),
                                    "candidate_title_batch": batch_candidates.copy(),
                                    "labels": batch_labels.copy(),
                                }
                                
                                # Reset batch index and clear arrays
                                batch_idx = 0
                                batch_labels.fill(0)
                                batch_users.fill(0)
                                batch_candidates.fill(0)
                                batch_history.fill(0)
                                
                                # Periodic cache cleanup to prevent memory overflow
                                self._periodic_cache_cleanup()

            except Exception as e:
                print(f"Error processing user {user_idx}: {e}")
                continue

        # Yield remaining data if any
        if batch_idx > 0:
            yield {
                "user_index_batch": batch_users[:batch_idx].copy(),
                "clicked_title_batch": batch_history[:batch_idx].copy(),
                "candidate_title_batch": batch_candidates[:batch_idx].copy(),
                "labels": batch_labels[:batch_idx].copy(),
            }

    def _get_cached_user_history(self, user_idx: int) -> np.ndarray:
        """
        Get user's historical news titles with caching for performance.
        
        Parameters:
        -----------
        user_idx : int
            User index
            
        Returns:
        --------
        np.ndarray
            User's historical news titles as word indices
        """
        if user_idx not in self.user_history_cache:
            # Get original user ID and their history
            raw_UID = self.user_idx2id[user_idx]
            raw_IID = self.user_history[raw_UID]
            
            # Process and cache the result
            self.user_history_cache[user_idx] = self.process_history_news_title(
                raw_IID, self.history_size
            )
            
        return self.user_history_cache[user_idx]

    def _fill_candidate_titles(self, batch_slot: np.ndarray, candidate_keys: List[int]) -> None:
        """
        Fill candidate news titles directly into pre-allocated array slot.
        
        Parameters:
        -----------
        batch_slot : np.ndarray
            Pre-allocated array slot to fill
        candidate_keys : list
            List of candidate item keys
        """
        try:
            # Convert candidate keys to raw item IDs
            raw_item_ids = [self.item_idx2id[k] for k in candidate_keys]
            
            # Fill each candidate title
            for i, raw_id in enumerate(raw_item_ids):
                if raw_id in self.news_index_map:
                    news_idx = self.news_index_map[raw_id]
                    batch_slot[i] = self.news_title_index[news_idx]
                else:
                    # Fill with zeros if news not found
                    batch_slot[i] = 0
                    
        except Exception as e:
            print(f"Error filling candidate titles: {e}")
            batch_slot.fill(0)

    def _periodic_cache_cleanup(self) -> None:
        """
        Periodically clean up caches to prevent memory overflow.
        """
        # Clean user history cache if it gets too large
        if len(self.user_history_cache) > self.max_cache_size:
            # Keep only the most recent half of the cache
            items = list(self.user_history_cache.items())
            self.user_history_cache = dict(items[len(items)//2:])
            
        # Clean news tokenization cache if it gets too large
        if len(self.news_tokenization_cache) > self.max_cache_size:
            items = list(self.news_tokenization_cache.items())
            self.news_tokenization_cache = dict(items[len(items)//2:])

    def _convert_data(self, label_list: List[List[int]], user_indexes: List[List[int]], 
                     candidate_title_indexes: List[np.ndarray], 
                     click_title_indexes: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert data lists into numpy arrays for model operation.
        
        Note: This method is kept for backward compatibility but is not used
        in the optimized batch generator.

        Parameters:
        -----------
        label_list : list
            List of ground-truth labels
        user_indexes : list
            List of user indexes
        candidate_title_indexes : list
            List of candidate news titles' word indices
        click_title_indexes : list
            List of word indices for user's clicked news titles

        Returns:
        --------
        dict
            Dictionary containing numpy arrays for model input
        """
        labels = np.asarray(label_list, dtype=np.float32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(candidate_title_indexes, dtype=np.int64)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        
        return {
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "labels": labels,
        }

    def map_news_titles_to_Cornac_internal_ids(self, train_set, news_original_id_to_news_title: Dict[Any, str]) -> Dict[int, str]:
        """
        Map news titles from original IDs to Cornac internal IDs.
        
        Parameters:
        -----------
        train_set : object
            Training dataset containing ID mappings
        news_original_id_to_news_title : dict
            Dictionary mapping original news IDs to news titles
            
        Returns:
        --------
        dict
            Dictionary mapping Cornac internal IDs to news titles
        """
        # Cache ID mappings
        self.item_id2idx = train_set.iid_map
        self.item_idx2id = {v: k for k, v in train_set.iid_map.items()}
        self.user_id2idx = train_set.uid_map
        self.user_idx2id = {v: k for k, v in train_set.uid_map.items()}
        
        # Create feature map with internal IDs
        feature_map = {}
        for key, value in news_original_id_to_news_title.items():
            if key in self.item_id2idx:
                idx = self.item_id2idx[key]
                feature_map[idx] = value

        # Check for missing keys and report
        missing_keys = set(self.item_id2idx.values()) - set(feature_map.keys())
        
        if not missing_keys:
            print("✓ All keys in item_id2idx are present in feature_map.")
        else:
            print(f"⚠ Missing keys in feature_map: {len(missing_keys)} items")
            if len(missing_keys) <= 10:  # Only print if not too many
                raw_ids = [self.item_idx2id[id0] for id0 in missing_keys]
                print(f"Missing raw item IDs: {raw_ids}")

        return feature_map

    def process_history_news_title(self, history_raw_IID: List[int], history_size: int) -> np.ndarray:
        """
        Process user's historical news titles into word index matrix.
        
        Parameters:
        -----------
        history_raw_IID : list
            List of raw item IDs from user's history
        history_size : int
            Fixed history size to maintain
            
        Returns:
        --------
        np.ndarray
            Matrix of word indices for historical news titles
        """
        def pad_or_truncate(sequence: List[int], max_length: int) -> List[int]:
            """Pad with -1 or truncate sequence to desired length."""
            if len(sequence) < max_length:
                return [-1] * (max_length - len(sequence)) + sequence
            else:
                return sequence[-max_length:]

        # Normalize history length
        history_raw_IID = pad_or_truncate(history_raw_IID, history_size)
        
        # Collect news titles for each item in history
        news_titles = []
        for item_id in history_raw_IID:
            if item_id in self.news_title:
                # Use cached tokenization if available
                if item_id not in self.news_tokenization_cache:
                    self.news_tokenization_cache[item_id] = self.word_tokenize(self.news_title[item_id])
                news_titles.append(self.news_tokenization_cache[item_id])
            elif item_id == -1:
                news_titles.append([])  # Empty title for padding
            else:
                news_titles.append([])  # Unknown item, treat as empty

        # Convert to word index matrix
        his_index = np.zeros((len(news_titles), self.title_size), dtype=np.int32)
        
        for i, title in enumerate(news_titles):
            for word_index in range(min(self.title_size, len(title))):
                word = title[word_index].lower()
                if word in self.word_dict:
                    his_index[i, word_index] = self.word_dict[word]
                    
        return his_index

    def init_news(self, news_title_json: Dict[Any, str]) -> None:
        """
        Initialize news information including news title indices.
        
        Parameters:
        -----------
        news_title_json : dict
            Dictionary mapping news IDs to news titles
        """
        print("Initializing news title indices...")
        
        # Create a copy and ensure we have empty title for -1 (padding)
        news_json = news_title_json.copy()
        news_json[-1] = ""
        
        # Create sequential index mapping for news
        self.news_index_map = {key: idx for idx, key in enumerate(news_json.keys())}
        
        # Tokenize all news titles and cache results
        news_title_tokens = {}
        for key, value in news_json.items():
            if key == -1:
                news_title_tokens[key] = []  # Empty for padding
            else:
                tokens = self.word_tokenize(value)
                news_title_tokens[key] = tokens
                # Cache tokenized version
                self.news_tokenization_cache[key] = tokens

        # Create word index matrix for all news
        self.news_title_index = np.zeros((len(news_title_tokens), self.title_size), dtype=np.int32)
        
        for key, title_tokens in news_title_tokens.items():
            mapped_index = self.news_index_map[key]
            for word_index in range(min(self.title_size, len(title_tokens))):
                word = title_tokens[word_index].lower()
                if word in self.word_dict:
                    self.news_title_index[mapped_index, word_index] = self.word_dict[word]
                    
        print(f"✓ Initialized {len(news_title_tokens)} news titles")

    def word_tokenize(self, sent: str) -> List[str]:
        """
        Split sentence into word list using regex.
        
        Parameters:
        -----------
        sent : str
            Input sentence
            
        Returns:
        --------
        list
            List of words/tokens
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []

    def clear_cache(self) -> None:
        """
        Clear all caches to free up memory.
        """
        self.user_history_cache.clear()
        self.news_tokenization_cache.clear()
        self.click_title_all_users.clear()
        
        # Force garbage collection
        gc.collect()
        print("✓ Cleared all caches")

    def optimize_memory_usage(self) -> None:
        """
        Optimize memory usage by adjusting cache sizes and cleaning up.
        """
        # Reduce cache sizes
        self.max_cache_size = 500
        self.batch_memory_limit = 8
        
        # Clean up large caches
        self._periodic_cache_cleanup()
        
        print(f"✓ Optimized memory usage - cache limit: {self.max_cache_size}, batch limit: {self.batch_memory_limit}")

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get current memory usage statistics.
        
        Returns:
        --------
        dict
            Dictionary with cache sizes and memory usage info
        """
        return {
            "user_history_cache_size": len(self.user_history_cache),
            "news_tokenization_cache_size": len(self.news_tokenization_cache),
            "click_title_cache_size": len(self.click_title_all_users),
            "max_cache_size": self.max_cache_size,
            "batch_memory_limit": self.batch_memory_limit
        }