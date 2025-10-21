# ML-Powered Cache Replacement SystemML-Cache-Project

================

A machine learning-based cache replacement policy that outperforms traditional LRU (Least Recently Used) by learning access patterns from real workloads.

This project demonstrates a learned cache eviction policy vs LRU.

## 🎯 Project Overview

Files:

This project implements an intelligent cache replacement system using **LightGBM** machine learning model to predict which cache blocks should be evicted. Trained on real **Azure LLM Inference Trace** data, it achieves **better hit rates** than traditional LRU across all cache sizes.- `1_feature_engineering.py` - reads `trace.txt` and produces `features.csv`.

- `2_train_model.py` - trains a LightGBM model and saves `cache_model.pkl`.

## 📊 Performance Results- `cache_simulator.py` - contains `LRUCache` and `LearnedCache` classes.

- `3_benchmark.py` - runs comparisons and prints final hit rates.

Tested on 100,000 augmented Azure LLM inference accesses:

Data:

| Cache Size | LRU Hit Rate | ML Hit Rate | Improvement |- `trace.txt` - NOT INCLUDED. You must download a large real-world trace (MSR Cambridge / SNIA / CloudPhysics) and place it here.

|------------|--------------|-------------|-------------|

| **100**    | 27.29%       | **28.23%**  | **+0.94%**  |Quick start (PowerShell):

| **500**    | 38.07%       | **41.52%**  | **+3.45%**  |

| **1000**   | 50.30%       | **54.00%**  | **+3.70%**  |```powershell

| **2000**   | 70.58%       | **72.33%**  | **+1.75%**  |python -m pip install -r requirements.txt

python .\1_feature_engineering.py 2000000    # generate features.csv from first 2M lines

🏆 **ML consistently beats LRU across all cache sizes!**python .\2_train_model.py                   # trains cache_model.pkl

python .\3_benchmark.py 2000000             # runs benchmark on last 20%

## 🚀 Features```



- **Enhanced Feature Engineering**: 8 sophisticated features including:Notes:

  - Recency & Frequency- For quick tests, pass a smaller number to the scripts (e.g., 20000).

  - Log-transformed features- The scripts perform basic checks and will exit with helpful messages if `trace.txt` or `features.csv` are missing.

  - Interval variance (pattern regularity)- The bottleneck is the trace file size; use the `nrows` argument to limit memory.

  - Average access interval

  - Block ageLicense: MIT

  - Recent access rate

- **Advanced ML Model**:
  - LightGBM with 500 trees
  - Deep trees (depth=8)
  - L1/L2 regularization
  - Feature bagging

- **Data Augmentation**:
  - Augments small datasets while preserving patterns
  - Maintains hot/warm/cold block distributions
  - Preserves temporal locality

## 📁 Project Structure

```
ML-Cache-Project/
│
├── 0_convert_azure_trace.py   # Convert Azure LLM trace to cache format
├── 0_augment_azure_trace.py   # Augment trace data for training
├── 0_generate_data.py          # Generate synthetic trace data
├── 1_feature_engineering.py    # Extract enhanced features
├── 2_train_model.py            # Train LightGBM model
├── 3_benchmark.py              # Compare ML vs LRU performance
├── 4_summary.py                # Generate results summary
├── cache_simulator.py          # LRU and ML cache implementations
│
├── AzureLLMInferenceTrace_code.csv  # Real Azure LLM trace data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/aithal007/ML-powered-Cache-Replacement-System.git
cd ML-powered-Cache-Replacement-System
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Option 1: Use Real Azure LLM Trace (Recommended)

```bash
# Step 1: Convert Azure trace to cache format
python 0_convert_azure_trace.py

# Step 2: Augment trace for more training data (optional but recommended)
python 0_augment_azure_trace.py

# Step 3: Extract features
python 1_feature_engineering.py

# Step 4: Train the ML model
python 2_train_model.py

# Step 5: Run benchmark
python 3_benchmark.py

# Step 6: View summary
python 4_summary.py
```

### Option 2: Generate Synthetic Data

```bash
# Step 1: Generate synthetic trace
python 0_generate_data.py

# Then follow steps 3-6 from Option 1
```

## 📚 Technical Details

### Feature Engineering

The system extracts 8 features for each cache access:

1. **Recency**: Time since last access
2. **Frequency**: Total access count
3. **Log Recency**: Log-transformed recency (handles skew)
4. **Log Frequency**: Log-transformed frequency
5. **Interval Variance**: Regularity of access pattern
6. **Average Interval**: Mean time between accesses
7. **Age**: Time since first access
8. **Recent Access Rate**: Accesses in recent window

### Model Architecture

- **Algorithm**: LightGBM Gradient Boosting
- **Trees**: 500 estimators
- **Max Depth**: 8
- **Learning Rate**: 0.05
- **Regularization**: L1=0.1, L2=0.1
- **Bagging**: 80% subsample, 80% feature sampling

### Cache Simulator

Two cache implementations:

1. **LRUCache**: Traditional Least Recently Used
2. **LearnedCache**: ML-powered with predictive eviction

## 📊 Dataset

The project uses the **Azure LLM Inference Trace** dataset containing:
- 8,819 real LLM inference requests
- 3,552 unique context token sizes
- Context range: 3-7,437 tokens
- Represents real KV-cache access patterns

Augmented to 100,000 accesses for robust training.

## 🔬 Research Background

This project demonstrates that machine learning can learn complex access patterns that simple heuristics like LRU miss. Key insights:

- **Temporal Locality**: ML learns burst patterns and periodic accesses
- **Frequency Patterns**: Distinguishes hot/warm/cold data better
- **Predictive Power**: Forecasts reuse distance more accurately
- **Adaptive**: Learns workload-specific patterns

## 📈 Future Improvements

- [ ] Add more cache policies (LFU, ARC, LIRS)
- [ ] Implement online learning for dynamic workloads
- [ ] Support for multi-tier caching
- [ ] GPU-accelerated inference
- [ ] Real-time adaptation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**aithal007**
- GitHub: [@aithal007](https://github.com/aithal007)

## 🙏 Acknowledgments

- Azure LLM Inference Trace dataset
- LightGBM library
- Research on learned cache replacement policies

---

**⭐ If you find this project useful, please consider giving it a star!**
