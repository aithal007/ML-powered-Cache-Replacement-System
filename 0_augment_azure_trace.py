"""
Augment the Azure LLM trace to create more training data.
This uses the real access patterns to generate synthetic but realistic data.
"""
import pandas as pd
import numpy as np

def augment_trace(input_file='AzureLLMInferenceTrace_code.csv', 
                  target_size=100000, 
                  output_file='trace.txt'):
    """
    Augment the Azure trace by:
    1. Learning temporal patterns
    2. Learning frequency distributions
    3. Generating synthetic accesses that follow the same patterns
    """
    print(f"Loading Azure trace from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} LLM inference requests")
    
    # Extract context tokens as blocks
    original_blocks = df['ContextTokens'].values
    unique_blocks = np.unique(original_blocks)
    
    print(f"Original: {len(original_blocks)} accesses, {len(unique_blocks)} unique blocks")
    
    # Analyze frequency distribution
    block_freq = pd.Series(original_blocks).value_counts()
    block_probs = (block_freq / block_freq.sum()).values
    
    # Identify hot, warm, and cold blocks
    hot_threshold = block_freq.quantile(0.8)  # Top 20%
    warm_threshold = block_freq.quantile(0.5)  # Middle 30%
    
    hot_blocks = block_freq[block_freq >= hot_threshold].index.tolist()
    warm_blocks = block_freq[(block_freq < hot_threshold) & 
                            (block_freq >= warm_threshold)].index.tolist()
    cold_blocks = block_freq[block_freq < warm_threshold].index.tolist()
    
    print(f"Hot blocks (top 20%): {len(hot_blocks)}")
    print(f"Warm blocks (middle 30%): {len(warm_blocks)}")
    print(f"Cold blocks (bottom 50%): {len(cold_blocks)}")
    
    # Generate augmented trace
    augmented_trace = list(original_blocks)  # Start with original
    
    # Parameters for realistic generation
    burst_prob = 0.25  # Probability of burst (repeated access)
    hot_prob = 0.6     # Probability of accessing hot set
    warm_prob = 0.25   # Probability of accessing warm set
    
    last_block = None
    
    print(f"Generating {target_size - len(original_blocks)} additional accesses...")
    
    for i in range(target_size - len(original_blocks)):
        # Burst pattern - repeat recent access
        if last_block is not None and np.random.random() < burst_prob:
            augmented_trace.append(last_block)
        # Hot block access
        elif np.random.random() < hot_prob and len(hot_blocks) > 0:
            last_block = np.random.choice(hot_blocks)
            augmented_trace.append(last_block)
        # Warm block access
        elif np.random.random() < warm_prob and len(warm_blocks) > 0:
            last_block = np.random.choice(warm_blocks)
            augmented_trace.append(last_block)
        # Cold block or random access
        else:
            if len(cold_blocks) > 0:
                last_block = np.random.choice(cold_blocks)
            else:
                last_block = np.random.choice(unique_blocks)
            augmented_trace.append(last_block)
    
    # Save augmented trace
    trace_df = pd.DataFrame({'block_id': augmented_trace})
    trace_df.to_csv(output_file, index=False, header=False)
    
    print(f"\n--- Augmented Trace Statistics ---")
    print(f"Total accesses: {len(augmented_trace)}")
    print(f"Unique blocks: {len(np.unique(augmented_trace))}")
    print(f"Saved to {output_file}")
    
    # Compare distributions
    print(f"\nTop 10 most frequent blocks:")
    top_aug = pd.Series(augmented_trace).value_counts().head(10)
    for block, count in top_aug.items():
        print(f"  {block} tokens: {count} times ({100*count/len(augmented_trace):.2f}%)")
    
    return trace_df


if __name__ == '__main__':
    augment_trace(target_size=100000)  # Generate 100k accesses
    print("\nNow run: python .\\1_feature_engineering.py")
