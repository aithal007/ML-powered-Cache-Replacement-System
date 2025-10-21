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
    MODEL_PATH = 'cache_model.pkl'

    results = []

    for capacity in cache_capacities:
        print(f"\n--- Testing Cache Capacity: {capacity} ---")
        lru_cache = LRUCache(capacity=capacity)
        learned_cache = LearnedCache(capacity=capacity, model_path=MODEL_PATH)

        print("Simulating LRU Cache...")
        for block_id in tqdm(test_trace, desc="LRU"):
            if lru_cache.get(block_id) == -1:
                lru_cache.put(block_id, block_id)

        print("Simulating Learned Cache...")
        for block_id in tqdm(test_trace, desc="Learned"):
            if learned_cache.get(block_id) == -1:
                learned_cache.put(block_id, block_id)

        lru_hit_rate = lru_cache.get_hit_rate()
        learned_hit_rate = learned_cache.get_hit_rate()

        results.append({
            'Capacity': capacity,
            'LRU Hit %': lru_hit_rate * 100,
            'Learned Hit %': learned_hit_rate * 100
        })

    print("\n--- FINAL BENCHMARK RESULTS ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
