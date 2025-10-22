from cache_simulator import LRUCache, LearnedCache
import pandas as pd
from tqdm import tqdm
import sys

DEFAULT_NROWS = 100000


def get_nrows():
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            print("Invalid nrows passed as argument. Using default.")
    return DEFAULT_NROWS


def main():
    nrows = get_nrows()

    try:
        full_trace_df = pd.read_csv(
            'trace.txt',
            header=None,
            names=['block_id'],
            nrows=nrows
        )
    except FileNotFoundError:
        print("Error: trace.txt not found. Please place the trace file in the project root.")
        sys.exit(1)

    split_point = int(len(full_trace_df) * 0.8)
    test_trace = full_trace_df.iloc[split_point:]['block_id'].tolist()
    print(f"Loaded {len(test_trace)} accesses for benchmark.")

    cache_capacities = [100, 500, 1000, 2000]
    LGBM_MODEL_PATH = 'cache_model_lgbm.pkl'
    RF_MODEL_PATH = 'cache_model_rf.pkl'
    LSTM_MODEL_PATH = 'cache_model_lstm.h5'

    results = []

    for capacity in cache_capacities:
        print(f"\n--- Testing Cache Capacity: {capacity} ---")
        lru_cache = LRUCache(capacity=capacity)
        learned_cache_lgbm = LearnedCache(capacity=capacity, model_path=LGBM_MODEL_PATH)
        learned_cache_rf = LearnedCache(capacity=capacity, model_path=RF_MODEL_PATH)
        learned_cache_lstm = LearnedCache(capacity=capacity, model_path=LSTM_MODEL_PATH)

        print("Simulating LRU Cache...")
        for block_id in tqdm(test_trace, desc="LRU"):
            if lru_cache.get(block_id) == -1:
                lru_cache.put(block_id, block_id)

        print("Simulating LightGBM Cache...")
        for block_id in tqdm(test_trace, desc="LightGBM"):
            if learned_cache_lgbm.get(block_id) == -1:
                learned_cache_lgbm.put(block_id, block_id)

        print("Simulating Random Forest Cache...")
        for block_id in tqdm(test_trace, desc="RandomForest"):
            if learned_cache_rf.get(block_id) == -1:
                learned_cache_rf.put(block_id, block_id)

        print("Simulating LSTM Cache...")
        for block_id in tqdm(test_trace, desc="LSTM"):
            if learned_cache_lstm.get(block_id) == -1:
                learned_cache_lstm.put(block_id, block_id)

        lru_hit_rate = lru_cache.get_hit_rate()
        lgbm_hit_rate = learned_cache_lgbm.get_hit_rate()
        rf_hit_rate = learned_cache_rf.get_hit_rate()
        lstm_hit_rate = learned_cache_lstm.get_hit_rate()

        results.append({
            'Capacity': capacity,
            'LRU Hit %': lru_hit_rate * 100,
            'LightGBM Hit %': lgbm_hit_rate * 100,
            'RandomForest Hit %': rf_hit_rate * 100,
            'LSTM Hit %': lstm_hit_rate * 100
        })

    print("\n--- FINAL BENCHMARK RESULTS ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
