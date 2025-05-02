# HySOM

**Fast, lightweight Python library for training Self-Organizing Maps on multidimensional time series, tailored for analyzing concentration-discharge hysteresis loops.**


## 🚀 Overview

HySOM is a Python library that simplifies training and visualizing Self-Organizing Maps (SOMs) for 2D time series analysis. It is specifically designed to study concentration–discharge (C–Q) hysteresis loops in streams leveraging [tslearn's Dynamic Time Warping implementation](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html) for the distance function, although other options are available such as Euclidean, Cosine and custom functions. HySOM also offers flexibility on the neighborhood and decay functions. See the [Documentation](www.documentation.com) for more details.

---

## 🔍 Features

- 🚵🏼 Easy training of rectangular **Self-Organizing Maps** (hexagonal lattices are not yet implemented)
- 📈 Tailored for **2-dimensional time series data**
- ➰ Supports the **Dynamic Time Warping** distance function 
- 🔄 Tools for analyzing and classifying C-Q **hysteresis loops**
- 🖼️ **Visualization** utilities for SOM grids and temporal trajectories
- 🔧 Lightweight and dependency-minimized

---


# 🌊 Use Case: C-Q Hysteresis Loop Analysis
Includes the General T–Q SOM, a standard framework for analyzing sediment transport hysteresis loops. Details on its development can be found [here](link.to.my.paper). Usage examples can be found in the [Documentation](www.documentation.com)

---

## 📦 Dependencies
HySOM requires the following libraries for proper functioning:  
- numpy
- tslearn
- matplotlib



---

### 🤝 Contributing
We welcome contributions! If you'd like to improve the code, report issues, or request features, please open a GitHub issue or pull request.


<!-- [Tutorial](https://colab.research.google.com/drive/1lNRfSmOkerxerLiB5Gw910OUH5XNzypw?usp=sharing) -->
