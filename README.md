# 🧠 ML-Powered Cache Replacement System

Machine learning models that **outperform traditional LRU** by learning access patterns from real Azure LLM workloads.

---

## 🎯 Results

Tested on **100,000 cache accesses** from Azure LLM Inference Trace:

| Model | Training RMSE | Avg Hit Rate | Improvement | Speed |
|-------|---------------|--------------|-------------|-------|
| **LSTM** 🏆 | **10,741** | **49.03%** | **+2.47%** | ~10 it/s |
| Random Forest | 10,768 | 49.03% | +2.47% | ~15 it/s |
| LightGBM | 11,386 | 49.02% | +2.46% | ~130 it/s |
| LRU (baseline) | - | 46.56% | - | Very Fast |

**Best Gain**: +3.75% at cache size 1000

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python 0_convert_azure_trace.py      # Convert trace
python 0_augment_azure_trace.py      # Augment 8k → 100k
python 1_feature_engineering.py      # Extract features
python 2_train_model.py              # Train all 3 models
python 3_benchmark.py                # Compare all 4 policies
```

---

## 🧬 How It Works

### Traditional LRU
- Uses **1 feature**: recency (time since last access)
- Simple, fast, but misses patterns

### ML Models (This Project)
- Uses **8 features**: recency, frequency, log-transforms, interval variance, avg interval, age, recent access rate
- **Learns patterns**: periodic access, burst detection, hot/warm/cold blocks
- **Predicts**: which items will be needed soon

### Models

**LSTM (Best Accuracy)**
- 2-layer LSTM network (128 + 64 units)
- Learns sequential patterns in cache access
- Best for: Maximum accuracy, temporal dependencies

**Random Forest**
- 500 decision trees, depth 8
- Robust ensemble learning
- Best for: Balanced accuracy & speed

**LightGBM (Fastest)**
- Gradient boosting, 500 trees
- Optimized for inference speed
- Best for: Speed-critical applications

---

## 📊 Dataset

**Azure LLM Inference Trace** - Real production workload
- Source: `AzureLLMInferenceTrace_code.csv`
- Original: 8,819 KV-cache requests
- Augmented: 100,000 accesses (preserves statistical properties)

---

## 🗂️ Project Structure

```
├── 0_convert_azure_trace.py    # Convert CSV to trace format
├── 0_augment_azure_trace.py    # Augment data 8k → 100k
├── 1_feature_engineering.py    # Extract 8 features
├── 2_train_model.py            # Train LightGBM + RF + LSTM
├── 3_benchmark.py              # Compare all 4 policies
├── cache_simulator.py          # LRU & ML cache implementations
└── requirements.txt            # Dependencies
```

---

## 📈 Performance by Cache Size

| Cache Size | LRU | LightGBM | Random Forest | LSTM | Best Improvement |
|------------|-----|----------|---------------|------|------------------|
| 100 | 27.29% | 28.23% | 28.22% | 28.22% | +0.94% |
| 500 | 38.07% | 41.52% | 41.48% | 41.48% | +3.45% |
| **1000** | 50.30% | 54.00% | 54.05% | 54.05% | **+3.75%** |
| 2000 | 70.58% | 72.33% | 72.36% | 72.36% | +1.78% |

---

## 🛠️ Dependencies

```txt
pandas
numpy
scikit-learn
lightgbm
tensorflow>=2.10.0
joblib
tqdm
```

---

## 🚀 Why ML Beats LRU

| Aspect | LRU | ML Models |
|--------|-----|-----------|
| Features | 1 (recency) | 8 (recency, frequency, patterns, etc.) |
| Learning | None | Learns workload-specific patterns |
| Pattern Detection | ❌ | ✅ Periodic, burst, hot/cold |
| Prediction | Reactive | Predictive |
| Accuracy | Baseline | +2.5% average improvement |

---

## 📄 License

MIT License

---

## 👤 Author

**aithal007** - [GitHub](https://github.com/aithal007)

---

**⭐ Star this repo if you find it useful!**
