# HySOM

**Fast, lightweight Python library for training Self-Organizing Maps on 2D time series, tailored for analyzing concentration-discharge hysteresis loops.**


## 🚀 Overview

**HySOM** is a Python library that simplifies the training and visualization of Self-Organizing Maps (SOMs) for 2D time series. It is specifically designed for the study of concentration–discharge (C–Q) hysteresis loops. With **HySOM**, you can access the **General T-Q SOM**—a standard framework for classifying sediment transport hysteresis loops. The library also includes several visualization tools to streamline the analysis of sediment transport hysteresis loops. Additionally, **HySOM** allows you to train your own SOM for C–Q analysis.

---

## 🔍 Features

- Direct access to the **General T-Q SOM** for sediment transport hysteresis loop analysis
- Tools for analyzing and classifying C-Q **hysteresis loops**
- **Visualization** utilities for SOM grids and hysteresis loops
- Easy, yet flexible, training of rectangular **Self-Organizing Maps** for 2D sequences
- Supports the **Dynamic Time Warping** distance function 
- Lightweight and dependency-minimized

---

# 🌊 The General T-Q SOM
Includes the General T–Q SOM, a standard framework for analyzing sediment transport hysteresis loops. Usage examples can be found in the [Documentation](https://hysom.readthedocs.io/en/latest/)

<p align="center">
  <img src="https://raw.githubusercontent.com/ArlexMR/HySOM/refs/heads/main/docs/images/generalTQsom.png" alt="General T-Q SOM" width="600">
</p>

---
# 📖 [Documentation](https://hysom.readthedocs.io/en/latest/)
Comprehensive documentation is provided, including quickstart tutorials, How-to guides and an API reference. [Click Here!](https://hysom.readthedocs.io/en/latest/)

---
## 📦 Dependencies
HySOM requires the following libraries for proper functioning (which are automatically installed when installing **HySOM**):  
- numpy
- tslearn
- matplotlib

---

### 🤝 Contributing
We welcome contributions! If you'd like to include your own standard SOM for C-Q hysteresis analysis, improve the code, report issues, or request features, please open a GitHub issue or pull request.


<!-- [Tutorial](https://colab.research.google.com/drive/1lNRfSmOkerxerLiB5Gw910OUH5XNzypw?usp=sharing) -->
