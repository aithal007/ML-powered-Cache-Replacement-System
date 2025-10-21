"""
Generate synthetic cache trace data based on real patterns.
This creates a larger dataset suitable for ML training.
"""
import pandas as pd
import numpy as np
import sys

def generate_synthetic_trace(base_trace, target_size=10000, output_file='trace.txt'):
    """
    Generate synthetic trace by analyzing patterns in base trace and creating
    realistic access patterns with temporal locality and frequency patterns.
    """
    print(f"Generating {target_size} synthetic accesses from {len(base_trace)} base patterns...")
    
    # Analyze the base trace
    unique_blocks = base_trace['block_id'].unique()
    block_frequencies = base_trace['block_id'].value_counts()
    
    # Normalize frequencies to probabilities
    block_probs = (block_frequencies / block_frequencies.sum()).values
    
    synthetic_trace = []
    
    # Generate with locality - simulate hot/cold blocks
    hot_set_size = min(len(unique_blocks) // 10, 100)  # Top 10% or max 100
    hot_blocks = block_frequencies.head(hot_set_size).index.tolist()
    
    # Parameters for access patterns
    burst_prob = 0.3  # Probability of burst access to same block
    hot_prob = 0.6    # Probability of accessing hot set
    
    last_block = None
    
    for i in range(target_size):
        if last_block is not None and np.random.random() < burst_prob:
            # Burst - repeat last block
            synthetic_trace.append(last_block)
        elif np.random.random() < hot_prob:
            # Access from hot set
            last_block = np.random.choice(hot_blocks)
            synthetic_trace.append(last_block)
        else:
            # Access from full distribution
            last_block = np.random.choice(unique_blocks, p=block_probs)
            synthetic_trace.append(last_block)
    
    # Save to file
    synthetic_df = pd.DataFrame({'block_id': synthetic_trace})
    synthetic_df.to_csv(output_file, index=False, header=False)
    
    print(f"Generated {len(synthetic_trace)} accesses")
    print(f"Unique blocks: {len(set(synthetic_trace))}")
    print(f"Saved to {output_file}")
    
    return synthetic_df


if __name__ == '__main__':
    # Read the sample trace
    try:
        sample_df = pd.read_csv('msr-cambridge1-sample.csv', header=None, names=['block_id'])
        print(f"Loaded {len(sample_df)} base samples from msr-cambridge1-sample.csv")
    except FileNotFoundError:
        print("Error: msr-cambridge1-sample.csv not found")
        print("Please ensure the sample file is in the project directory")
        sys.exit(1)
    
    # Generate 500,000 synthetic accesses (5 lakh)
    target_size = 500000
    if len(sys.argv) > 1:
        try:
            target_size = int(sys.argv[1])
        except ValueError:
            pass
    
    generate_synthetic_trace(sample_df, target_size=target_size)
    print("\nNow run: python .\\1_feature_engineering.py")
