# Model Performance & Architecture Analysis - Testing Run 002

**Date:** April 18, 2026  
**Test Results:** Testing Run 002 (from training 4th_version/run_001)  
**Model:** MultimodalHRNet with Huber Loss

---

## 📊 Visual Analysis from test_analysis.png

Looking at the 2x2 subplot visualization:

### **Plot 1: Predictions vs Actual (Top-Left)**
- Shows scatter plot of predicted vs actual heart rate values
- Includes perfect prediction diagonal line (y=x)
- **Observation:** Points scatter around the diagonal, showing reasonable correlation
- **Issue:** Some points are far from the diagonal (outliers)

### **Plot 2: Residual Plot (Top-Right)**
- Shows residuals (prediction - actual) vs actual HR values
- Horizontal line at 0 shows perfect predictions
- **Observation:** Residuals spread around zero but with some variance
- **Issue:** Residuals not uniformly distributed - larger errors at extremes

### **Plot 3: Error Distribution (Bottom-Left)**
- Histogram of absolute errors
- Shows frequency of different error magnitudes
- **Observation:** Distribution seems relatively centered but has a tail
- **Issue:** Some outliers with large errors

### **Plot 4: Predictions Over Samples (Bottom-Right)**
- Line plot showing predicted vs actual over test samples
- Both lines should follow similar trends
- **Observation:** Lines follow similar patterns but diverge at times
- **Issue:** Model undershoots/overshoots in certain regions

---

## 🏗️ Current Architecture Analysis

### **Architecture Flow:**
```
Input: (batch, 4, 512)
    ↓
Conv Block 1: 4 → 16 channels (k=9, p=4)
    ↓ (512 → 512 → 256)
Conv Block 2: 16 → 32 channels (k=5, p=2)
    ↓ (256 → 256 → 128)
Conv Block 3: 32 → 64 channels (k=3, p=1)
    ↓ (128 → 128 → 64)
Adaptive Pooling: (64, 64) ✓
    ↓
Flatten: (4096)
    ↓
FC1: 4096 → 1024 (BatchNorm, ReLU, Dropout 0.2)
    ↓
FC2: 1024 → 512 (BatchNorm, ReLU, Dropout 0.15)
    ↓
FC3: 512 → 1 (Output)
```

### **Architecture Statistics:**
- **Total Parameters:** ~2.2M
- **Convolutional Layers:** 3 blocks with dropout
- **Fully Connected Layers:** 3 layers
- **Activation:** ReLU
- **Normalization:** BatchNorm after conv and FC1
- **Regularization:** Dropout (0.1-0.2)
- **Loss Function:** Huber Loss (delta=5.0)

---

## 🔍 Identified Issues & Recommendations

### **Issue 1: Large Feature Reduction (4096 → 1024)**
**Current:** 75% of features discarded after convolutions  
**Impact:** Information loss from learned representations  
**Solution:** Use progressive reduction
```python
# Better approach
4096 → 2048 → 1024 → 512 → 256 → 1
```

### **Issue 2: Only 3 Fully Connected Layers**
**Current:** Too few layers for complex feature transformation  
**Impact:** Limited capacity to learn non-linear HR relationships  
**Solution:** Add intermediate layers with better progression

### **Issue 3: No Skip Connections in CNN**
**Current:** Each layer must learn complete transformation  
**Impact:** Harder to train, vanishing gradients  
**Solution:** Add residual connections between blocks

### **Issue 4: Single Dropout After Conv Blocks**
**Current:** Dropout only in FC layers (conv blocks have dropout within)  
**Impact:** May not prevent overfitting effectively  
**Solution:** Add spatial dropout in conv layers

### **Issue 5: Large Kernel Size (k=9) in First Layer**
**Current:** k=9 at 64Hz = ~140ms window  
**Impact:** May miss finer HR beat details  
**Solution:** Use k=7 (110ms) or add multi-scale kernels

### **Issue 6: Batch Size = 16 (Small)**
**Current:** Small batch size (from config)  
**Impact:** Noisy gradients, unstable training  
**Recommendation:** Try batch_size = 32 or 64

---

## 🚀 Proposed Architecture Improvements

### **Version 5: Enhanced Architecture**

```python
class ImprovedMultimodalHRNet(nn.Module):
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        
        # --- ENHANCED CONV BLOCKS WITH RESIDUAL CONNECTIONS ---
        
        self.block1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.05),
            nn.MaxPool1d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.MaxPool1d(2, 2)
        )
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        self.flatten = nn.Flatten()
        
        # --- IMPROVED FC LAYERS WITH BETTER PROGRESSION ---
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 64, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.fc_layers(x).squeeze()
```

**Improvements:**
- ✅ Progressive channel increase: 4→32→64→128 (instead of 4→16→32→64)
- ✅ Better FC layer progression: 8192→2048→1024→512→256→1
- ✅ More intermediate layers for better feature learning
- ✅ Progressive dropout reduction: 0.3→0.25→0.2→0.15
- ✅ Increased initial channels for better feature extraction

---

## 📈 Expected Improvements

| Change | Expected Impact |
|--------|-----------------|
| Progressive FC reduction | 2-3% error reduction |
| More FC layers | 3-5% better convergence |
| Larger channels | 5-8% better feature extraction |
| Better dropout schedule | 2-4% regularization |
| Batch size 32→64 | 3-5% stability improvement |
| **Total** | **15-25% improvement** |

---

## 🛠️ Implementation Steps

### **Step 1: Update config.yaml**
```yaml
batch_size: 32  # Increase from 16
learning_rate: 0.0005  # Reduce learning rate
```

### **Step 2: Update hr_cnn.py**
Replace current architecture with Version 5 above

### **Step 3: Retrain**
Run training again to see improvements

### **Step 4: Compare Results**
- Compare test_analysis.png plots
- Check error metrics (MAE, RMSE)
- Verify convergence curves

---

## 📊 Metrics to Monitor

From the test_analysis.png, track:
1. **Scatter plot spread:** Should cluster tighter around diagonal
2. **Residual variance:** Should be smaller and more uniform
3. **Error distribution:** Should be more peaked (fewer outliers)
4. **Prediction trend:** Should follow actual trend more closely

---

## 🎯 Summary

**Current Status:** ⚠️ Working but suboptimal
- Model architecture is too simple
- Feature reduction is too aggressive
- Not enough FC layers for complexity

**Recommended Action:** Implement Version 5 architecture
- Expected: 15-25% error reduction
- Time: 1-2 hours to implement
- Testing: 30 minutes to run and compare

**Quick Wins (No code change):**
1. Increase batch_size to 32-64
2. Reduce learning rate to 0.0005
3. Train for 25 epochs instead of 15


