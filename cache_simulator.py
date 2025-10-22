import collections
import joblib
import numpy as np
import os

# Try to import TensorFlow for LSTM support
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

#################################################################
# CLASS 1: THE BASELINE CACHE
#################################################################
class LRUCache:
    """
    Implements a classic LRU (Least Recently Used) cache.
    We use collections.OrderedDict to make this fast and simple.
    It's an "ordered" dictionary, so it remembers the order
    items were inserted.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int):
        """
        Attempts to get a key from the cache.
        If it's a hit, we mark it as "most recently used."
        """
        if key not in self.cache:
            self.misses += 1
            return -1  # Signal a miss
        
        # KEY HIT!
        self.hits += 1
        # move_to_end() marks this key as the most recently used.
        self.cache.move_to_end(key)
        return self.cache[key] # Return the value

    def put(self, key: int, value: int):
        """
        Adds a new key-value pair to the cache.
        If the cache is full, it evicts the LEAST recently used item.
        """
        if key in self.cache:
            # If key already exists, just update it and mark as recent.
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # Check if we are over capacity
        if len(self.cache) > self.capacity:
            # popitem(last=False) removes the FIRST item added
            # (which is the Least Recently Used one).
            self.cache.popitem(last=False)

    def get_hit_rate(self):
        total_accesses = self.hits + self.misses
        return self.hits / total_accesses if total_accesses > 0 else 0

#################################################################
# CLASS 2: THE ML-POWERED CACHE
#################################################################
class LearnedCache:
    """
    Implements a cache that uses a pre-trained ML model to decide
    which item to evict.
    """
    def __init__(self, capacity: int, model_path: str):
        self.capacity = capacity
        self.cache = {} # A simple dict to store key -> value
        
        # Detect model type and load accordingly
        self.is_lstm = model_path.endswith('.h5') or model_path.endswith('.keras')
        
        # Load the trained model from the file
        try:
            if self.is_lstm:
                if not TENSORFLOW_AVAILABLE:
                    raise ImportError("TensorFlow is not installed. Install it with: pip install tensorflow")
                self.model = keras.models.load_model(model_path)
                # Load the scaler for LSTM
                scaler_path = model_path.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded LSTM model from {model_path}")
            else:
                self.model = joblib.load(model_path)
                self.scaler = None
                print(f"Loaded {model_path.split('_')[-1].replace('.pkl', '')} model from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            print("Please run 2_train_model.py first.")
            raise
            
        self.hits = 0
        self.misses = 0
        
        # --- State for feature engineering ---
        # We must track the same features our model was trained on.
        self.current_time_step = 0
        self.last_seen = {}
        self.frequency = {}
        self.access_history = {}  # Track access times for enhanced features
        self.first_access = {}  # Track first access time

    def _update_access_stats(self, key):
        """Internal helper to update our feature-tracking state."""
        self.current_time_step += 1
        
        # Update frequency
        self.frequency[key] = self.frequency.get(key, 0) + 1
        # Update last seen time
        self.last_seen[key] = self.current_time_step
        # Track access history
        if key not in self.access_history:
            self.access_history[key] = []
            self.first_access[key] = self.current_time_step
        self.access_history[key].append(self.current_time_step)

    def get(self, key: int):
        """
        Attempts to get a key from the cache.
        We must update our stats on every access (hit or miss).
        """
        # We update stats *first*
        self._update_access_stats(key)

        if key not in self.cache:
            self.misses += 1
            return -1 # Signal a miss

        # KEY HIT!
        self.hits += 1
        return self.cache[key] # Return the value

    def put(self, key: int, value: int):
        """
        Adds a new key-value pair to the cache.
        If the cache is full, it calls the ML-driven evict() method.
        """
        # If the cache is full AND this is a new item, we must evict.
        if len(self.cache) >= self.capacity and key not in self.cache:
            self.evict()
        
        self.cache[key] = value

    def evict(self):
        """
        This is the "brains" of the operation.
        1. Get all items currently in the cache.
        2. Generate features for each item *right now*.
        3. Ask the ML model to predict their reuse distance.
        4. Evict the item with the HIGHEST (worst) predicted score.
        """
        if not self.cache:
            return # Nothing to evict

        cached_items = list(self.cache.keys())
        features_for_prediction = []

        # 1. & 2. Generate enhanced features for all cached items
        for item_key in cached_items:
            recency = self.current_time_step - self.last_seen.get(item_key, 0)
            freq = self.frequency.get(item_key, 0)
            
            # Enhanced features
            log_recency = np.log1p(recency)
            log_frequency = np.log1p(freq)
            
            # Interval variance and average
            if item_key in self.access_history and len(self.access_history[item_key]) >= 3:
                intervals = [self.access_history[item_key][j] - self.access_history[item_key][j-1] 
                           for j in range(1, len(self.access_history[item_key]))]
                interval_variance = np.var(intervals) if len(intervals) > 1 else 0
                avg_interval = np.mean(intervals)
            else:
                interval_variance = 0
                avg_interval = recency if recency > 0 else 1
            
            # Age
            age = self.current_time_step - self.first_access.get(item_key, self.current_time_step)
            
            # Recent access rate
            window_size = min(100, self.current_time_step)
            recent_accesses = sum(1 for t in self.access_history.get(item_key, []) 
                                if self.current_time_step - t <= window_size)
            
            # This MUST match the feature order from training!
            features_for_prediction.append([
                recency, freq, log_recency, log_frequency,
                interval_variance, avg_interval, age, recent_accesses
            ]) 
        
        # 3. Ask the model
        # We get an array of predicted reuse distances
        if self.is_lstm:
            # For LSTM: scale, reshape, and predict
            features_array = np.array(features_for_prediction)
            features_scaled = self.scaler.transform(features_array)
            features_reshaped = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
            predictions = self.model.predict(features_reshaped, verbose=0).flatten()
        else:
            # For tree-based models (LightGBM, Random Forest)
            predictions = self.model.predict(features_for_prediction)
        
        # 4. Find and evict the worst item
        # np.argmax() finds the index of the highest value
        eviction_candidate_index = np.argmax(predictions)
        key_to_evict = cached_items[eviction_candidate_index]
        
        del self.cache[key_to_evict]

    def get_hit_rate(self):
        total_accesses = self.hits + self.misses
        return self.hits / total_accesses if total_accesses > 0 else 0
