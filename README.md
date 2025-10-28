# 📊 Student Performance Analysis using Clustering & Classification

This project analyzes student exam performance using a combination of **Unsupervised Learning** (Clustering) and **Supervised Learning** (Classification).  
The aim is to uncover hidden performance patterns and predict whether a student will **pass Mathematics** using demographic and academic features.

---

## ✅ Key Objectives

- Perform **data preprocessing** & **feature encoding**
- Apply **Standard Scaling** for distance-based algorithms
- Implement **3 Clustering Algorithms**:
  - K-Means
  - Hierarchical Clustering (Ward Linkage + Dendrogram)
  - Gaussian Mixture Model (GMM)
- Evaluate clusters using:
  - Silhouette Score
  - Davies–Bouldin Index
  - Calinski–Harabasz Score
- Add cluster labels as new features to enhance prediction
- Train & compare **Classification Models**:
  - Bayesian Logistic Regression
  - SVM with RBF Kernel
- Visualize clusters using **PCA (2D Projection)**
- Save trained models using **Joblib**

---

## 🧠 Machine Learning Workflow

flowchart TD
A[Load Dataset] --> B[Feature Encoding & Cleaning]
B --> C[Standard Scaling]
C --> D[Clustering Algorithms: K-Means | H-Clustering | GMM]
D --> E[Cluster Validation Metrics]
E --> F[Add Cluster Features to Dataset]
F --> G[Train Classifiers: Logistic Regression & SVM]
G --> H[Evaluate Model Performance]
H --> I[Save Models with Joblib]
