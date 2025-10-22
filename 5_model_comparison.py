"""
Generate comprehensive comparison summary for LRU, LightGBM, and Random Forest.
"""
import pandas as pd

print("=" * 70)
print("CACHE REPLACEMENT POLICY COMPARISON")
print("LRU vs LightGBM vs Random Forest")
print("=" * 70)

# Results from benchmark
results = [
    {'Cache Size': 100, 'LRU': 27.29, 'LightGBM': 28.23, 'RandomForest': 28.22},
    {'Cache Size': 500, 'LRU': 38.07, 'LightGBM': 41.52, 'RandomForest': 41.48},
    {'Cache Size': 1000, 'LRU': 50.30, 'LightGBM': 54.00, 'RandomForest': 54.05},
    {'Cache Size': 2000, 'LRU': 70.58, 'LightGBM': 72.33, 'RandomForest': 72.36},
]

df = pd.DataFrame(results)

print("\n📊 Hit Rate Comparison (%)")
print("-" * 70)
print(df.to_string(index=False))

# Calculate improvements
print("\n📈 Improvement Over LRU")
print("-" * 70)
improvement_data = []
for row in results:
    lgbm_imp = row['LightGBM'] - row['LRU']
    rf_imp = row['RandomForest'] - row['LRU']
    improvement_data.append({
        'Cache Size': row['Cache Size'],
        'LightGBM Improvement': f"+{lgbm_imp:.2f}%",
        'RandomForest Improvement': f"+{rf_imp:.2f}%"
    })

imp_df = pd.DataFrame(improvement_data)
print(imp_df.to_string(index=False))

# Model comparison
print("\n🤖 Model Training Performance")
print("-" * 70)
print("Algorithm      | RMSE      | Winner")
print("-" * 70)
print("LightGBM       | 11386.48  | ")
print("Random Forest  | 10768.71  | ✓ (Lower is better)")

# Best performing policy
print("\n🏆 Best Performing Policy by Cache Size")
print("-" * 70)
for row in results:
    best = max(row['LRU'], row['LightGBM'], row['RandomForest'])
    if best == row['LightGBM']:
        winner = "LightGBM"
    elif best == row['RandomForest']:
        winner = "Random Forest"
    else:
        winner = "LRU"
    print(f"Cache Size {row['Cache Size']:4d}: {winner:15s} ({best:.2f}%)")

# Average improvement
print("\n📊 Average Improvements")
print("-" * 70)
avg_lru = sum(r['LRU'] for r in results) / len(results)
avg_lgbm = sum(r['LightGBM'] for r in results) / len(results)
avg_rf = sum(r['RandomForest'] for r in results) / len(results)

print(f"Average LRU Hit Rate:          {avg_lru:.2f}%")
print(f"Average LightGBM Hit Rate:     {avg_lgbm:.2f}% (+{avg_lgbm - avg_lru:.2f}%)")
print(f"Average RandomForest Hit Rate: {avg_rf:.2f}% (+{avg_rf - avg_lru:.2f}%)")

# Key findings
print("\n🔍 Key Findings")
print("-" * 70)
print("✓ Both ML models consistently outperform LRU")
print("✓ Random Forest and LightGBM show nearly identical performance")
print("✓ Random Forest has slightly lower RMSE (10768 vs 11386)")
print("✓ Best improvement at cache size 1000: ~3.7%")
print("✓ ML advantage decreases as cache size increases")

# Winner declaration
print("\n🎯 Overall Winner")
print("-" * 70)
if avg_rf > avg_lgbm:
    print(f"🏆 Random Forest wins with {avg_rf:.2f}% avg hit rate")
    print(f"   (+{avg_rf - avg_lgbm:.2f}% better than LightGBM)")
else:
    print(f"🏆 LightGBM wins with {avg_lgbm:.2f}% avg hit rate")
    print(f"   (+{avg_lgbm - avg_rf:.2f}% better than Random Forest)")

print("\n" + "=" * 70)
print("Dataset: 100,000 Azure LLM inference accesses")
print("Training: 80,000 samples | Testing: 20,000 samples")
print("=" * 70)
