
---

#  Machine Learning Mini-Projects: DBSCAN & PCA

This repository contains two small yet powerful experiments in **unsupervised learning** using `scikit-learn`.  
Each experiment focuses on a different aspect of data understanding:  
- **DBSCAN** for clustering and outlier detection  
- **PCA** for dimensionality reduction and data visualization  

---

##  Project Overview

###  DBSCAN Clustering & Outlier Detection

**Goal:**  
Detect clusters and noise points (outliers) in unlabelled data using the **DBSCAN algorithm**.

**Key Concepts:**
- **eps:** The maximum distance between two samples to be considered neighbors.  
- **min_samples:** Minimum number of neighbors required to form a dense region (a cluster).  
- **Outliers:** Points not belonging to any dense region are labeled as `-1`.

**Workflow:**
1. Generate or load a dataset.
2. Scale features using `StandardScaler`.
3. Apply `DBSCAN(eps=0.5, min_samples=5)`.
4. Visualize clusters and outliers in 2D.
5. Evaluate the number of clusters and noise points.

**Expected Outcome:**
- Clusters of varying shapes are detected automatically.
- Outliers are separated clearly (shown in different color).

---

###  PCA – Principal Component Analysis

**Goal:**  
Reduce the number of features while preserving as much variance as possible, and visualize high-dimensional data in 2D.

**Key Concepts:**
- **Explained Variance Ratio:** Shows how much information (variance) each principal component holds.  
- **Cumulative Variance:** Helps decide the number of components to keep (e.g., 95% of total variance).  
- **Reconstruction Error (MSE):** Measures data loss after dimensionality reduction.

**Workflow:**
1. Standardize features using `StandardScaler`.
2. Fit PCA with `n_components=None` to see variance ratios.
3. Plot both individual and cumulative explained variance.
4. Choose components where cumulative variance ≥ 95%.
5. Transform data → `fit_transform(X_scaled)`.
6. Reconstruct data → `inverse_transform(X_pca)`.
7. Compute **MSE** between original and reconstructed data.

**Expected Outcome:**
- Most variance captured by the first few components.  
- MSE should be low (indicating little information loss).  
- Reduced data visualized in 2D scatter plot.

---

##  Technologies Used

| Library | Purpose |
|----------|----------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `scikit-learn` | Machine learning (DBSCAN, PCA, StandardScaler) |

---


##  Results Summary

| Task | Main Observation |
|------|------------------|
| DBSCAN | Automatically found clusters and detected outliers with minimal tuning. |
| PCA | Reduced 3D data to 2D while retaining ~95% variance and minimal reconstruction error. |

---

##  Learnings

- DBSCAN is ideal for **non-linear, irregular clusters** and **outlier detection** without predefining `k`.  
- PCA helps in **noise reduction, visualization, and feature compression** while maintaining most of the data’s structure.  
- Standardization (`StandardScaler`) is essential before both algorithms.

---


##  Author

**Nau Raa**
 Focused on data science, AI learning, and practical understanding of core ML algorithms.

--

---

