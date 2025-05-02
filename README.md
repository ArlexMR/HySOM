# HySOM

**Fast, lightweight library for training Self-Organizing Maps on multidimensional time series, tailored for analyzing concentration-discharge hysteresis loops.**


## 🚀 Overview

HySOM is a Python library that streamlines the training and visualization of Self-Organizing Maps (SOMs) for multidimensional time series analysis. It is specifically designed for studying concentration–discharge (C–Q) hysteresis loops in streams—an essential tool for understanding the sources and transport pathways of suspended and dissolved constituents in freshwater systems. See the [Documentation] for more details

---

## 🔍 Features

- 🚵🏼 Easy training of 2D **Self-Organizing Maps**
- 📈 Tailored for **2-dimensional time series data**
- ➰ Supports the **Dynamic Time Warping** distance function 
- 🔄 Tools for analyzing and classifying C-Q **hysteresis loops**
- 🖼️ **Visualization** utilities for SOM grids and temporal trajectories
- 🔧 Lightweight and dependency-minimized

---

## 📦 Installation

### PyPI
```bash
pip install hysom
```

### conda

```bash
conda install -c conda-forge hysom
```
---
✨ Quick Example

```python
from hysom import SOM
import numpy as np

# Generate synthetic data
data = np.random.rand(100, 100, 2)  # representing 100 samples of 2D sequences, each with 100 time steps

# Train SOM
som = SOM(width=10, height=10, input_dim = data.shape[1:])
som.train(data, epochs = 5)

# Visualize results
plot_som_map(som)
```

# 🌊 Use Case: C-Q Hysteresis Loop Analysis
Includes the General T–Q SOM, a standard framework for analyzing sediment transport hysteresis loops. Details on its development can be found [here](link.to.my.paper). Usage examples can be found in the [Documentation](link.to.documentation.Generaltqmap)

---

### 📚 Documentation  
Comprehensive documentation is available at:  
👉 https://your-documentation-url

### 🤝 Contributing
We welcome contributions! If you'd like to improve the code, report issues, or request features, please open a GitHub issue or pull request.

### 🔗 Links
GitHub: https://github.com/yourusername/som-loop

PyPI: https://pypi.org/project/som-loop

Conda-Forge (if published): https://anaconda.org/conda-forge/som-loop


[Tutotial](https://colab.research.google.com/drive/1lNRfSmOkerxerLiB5Gw910OUH5XNzypw?usp=sharing)
