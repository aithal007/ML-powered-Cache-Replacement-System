import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys

# Allow limiting rows via env var or command-line arg for faster tests
DEFAULT_NROWS = None  # Use all available data

def get_nrows():
    # Priority: CLI arg > ENV var > default
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            print("Invalid nrows passed as argument. Using default.")
    env = os.environ.get('ML_CACHE_NROWS')
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return DEFAULT_NROWS


def main():
    print("Starting feature engineering...")

    nrows = get_nrows()

    # --- 1. Load the Trace Data ---
    try:
        trace_df = pd.read_csv(
            'trace.txt',
            header=None,
            names=['block_id'],
            nrows=nrows
        )
        trace = trace_df['block_id'].tolist()
    except FileNotFoundError:
        print("Error: trace.txt not found in the current folder.")
        print("Please download a real trace (MSR/SNIA/CloudPhysics), rename it to trace.txt, and place it here.")
        sys.exit(1)

    if len(trace) == 0:
        print("trace.txt is empty. Please provide a valid trace file.")
        sys.exit(1)

    print(f"Loaded {len(trace)} accesses from trace.txt")

    # --- 2. Calculate Labels (Reuse Distance) ---
    next_access = {}
    reuse_distances = [0] * len(trace)

    print("Calculating reuse distances (labels)...")
    # Iterate backwards through the trace
    for i in range(len(trace) - 1, -1, -1):
        block_id = trace[i]

        if block_id in next_access:
            distance = next_access[block_id] - i
            reuse_distances[i] = distance
        else:
            reuse_distances[i] = len(trace) * 2  # large "infinity"-ish value

        next_access[block_id] = i

    print("Done calculating labels.")

    # --- 3. Calculate Enhanced Features ---
    features = []
    last_seen = {}
    frequency = defaultdict(int)
    access_history = defaultdict(list)  # Track all access times
    
    print("Calculating enhanced features (recency, frequency, patterns)...")
    # Iterate forwards through the trace
    for i, block_id in enumerate(trace):
        # Feature 1: Recency (time since last access)
        recency = i - last_seen.get(block_id, -len(trace))
        
        # Feature 2: Frequency (how many times seen so far)
        frequency[block_id] += 1
        freq = frequency[block_id]
        
        # Feature 3: Access interval variance (regularity of access)
        access_history[block_id].append(i)
        if len(access_history[block_id]) >= 3:
            intervals = [access_history[block_id][j] - access_history[block_id][j-1] 
                        for j in range(1, len(access_history[block_id]))]
            interval_variance = np.var(intervals) if len(intervals) > 1 else 0
            avg_interval = np.mean(intervals)
        else:
            interval_variance = 0
            avg_interval = recency if recency > 0 else len(trace)
        
        # Feature 4: Time since first access (age)
        first_access = access_history[block_id][0]
        age = i - first_access
        
        # Feature 5: Recent access rate (accesses in last window)
        window_size = min(100, i)
        recent_accesses = sum(1 for t in access_history[block_id] if i - t <= window_size)
        
        # Feature 6: Logarithmic features (handle skewed distributions)
        log_recency = np.log1p(recency)
        log_frequency = np.log1p(freq)
        
        features.append({
            'recency': recency,
            'frequency': freq,
            'log_recency': log_recency,
            'log_frequency': log_frequency,
            'interval_variance': interval_variance,
            'avg_interval': avg_interval,
            'age': age,
            'recent_access_rate': recent_accesses
        })
        
        last_seen[block_id] = i

    print("Done calculating enhanced features.")

    # --- 4. Create and Save the DataFrame ---
    df = pd.DataFrame(features)
    df['reuse_distance'] = reuse_distances

    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]

    train_df.to_csv('features.csv', index=False)
    print(f"Saved {len(train_df)} rows of training data to features.csv")
    print("Feature engineering complete.")


if __name__ == '__main__':
    main()
