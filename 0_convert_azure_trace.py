"""
Convert Azure LLM Inference Trace to cache block format.
We'll use ContextTokens as the "block_id" since that represents
the code/context being accessed in the KV-cache.
"""
import pandas as pd
import numpy as np

def convert_azure_trace(input_file='AzureLLMInferenceTrace_code.csv', output_file='trace.txt'):
    """
    Convert Azure trace to simple block_id format.
    Uses ContextTokens as block identifiers (representing cached context).
    """
    print(f"Loading Azure trace from {input_file}...")
    
    # Load the trace
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} LLM inference requests")
    
    # Use ContextTokens as block_id (representing KV-cache blocks)
    # This simulates caching of context tokens
    block_ids = df['ContextTokens'].values
    
    print(f"Unique context sizes: {len(np.unique(block_ids))}")
    print(f"Context token range: {block_ids.min()} to {block_ids.max()}")
    
    # Save as trace.txt
    trace_df = pd.DataFrame({'block_id': block_ids})
    trace_df.to_csv(output_file, index=False, header=False)
    
    print(f"Saved {len(trace_df)} accesses to {output_file}")
    
    # Print some statistics
    print("\n--- Trace Statistics ---")
    print(f"Total accesses: {len(block_ids)}")
    print(f"Unique blocks: {len(np.unique(block_ids))}")
    print(f"Most frequent context sizes:")
    top_contexts = pd.Series(block_ids).value_counts().head(10)
    for ctx, count in top_contexts.items():
        print(f"  {ctx} tokens: {count} times ({100*count/len(block_ids):.2f}%)")
    
    return trace_df


if __name__ == '__main__':
    convert_azure_trace()
    print("\nNow run: python .\\1_feature_engineering.py")
