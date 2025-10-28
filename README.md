📊 Student Performance Analysis using Clustering & Classification

This project analyzes student exam performance using a combination of unsupervised learning (clustering) and supervised learning (classification). The goal is to understand hidden performance patterns and predict whether a student will pass in Mathematics based on demographic and academic attributes.

✅ Key Objectives

Perform data preprocessing and feature encoding

Apply Standard Scaling for distance-based algorithms

Use three clustering techniques:

K-Means

Hierarchical Clustering (with Dendrogram visualization)

Gaussian Mixture Model (GMM)

Evaluate cluster quality using:

Silhouette Score

Davies-Bouldin Index

Calinski-Harabasz Score

Integrate cluster labels as new features for prediction

Train and compare classification models:

Bayesian Logistic Regression

SVM with RBF Kernel

Visualize cluster separation using PCA 2D projection

Save trained models using Joblib

🧠 Machine Learning Workflow
flowchart TD
A[Load Dataset] --> B[Feature Encoding & Cleaning]
B --> C[Standard Scaling]
C --> D[Clustering Algorithms]
D --> E[Cluster Validation Metrics]
E --> F[Add Cluster Features to Dataset]
F --> G[Train Classifiers]
G --> H[Evaluate Model Performance]
H --> I[Save Models]

📂 Dataset

The dataset used: exams.csv

Includes:

Gender

Race/Ethnicity

Parental Education

Lunch Type

Test Preparation Course

Math / Reading / Writing Scores

Target Variable:
pass_math = 1 if Math ≥ 50 else 0

📌 Results Snapshot
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Bayesian Logistic Regression	✅ Varies on data				
SVM (RBF Kernel)	✅ Varies on data				

✔️ Clustering-enhanced features improved model performance

📈 Visualizations

✔️ Hierarchical Clustering Dendrogram
✔️ PCA-based Cluster Projection
✔️ Cluster metrics for quality assessment

All visual outputs included in notebook execution.

🛠️ Technologies Used
Category	Tools / Libraries
Language	Python
ML & Evaluation	scikit-learn, GaussianMixture, KMeans, AgglomerativeClustering
Visualization	Matplotlib, PCA, Dendrogram
Model Saving	Joblib
Data Handling	Pandas, NumPy
✅ How to Run
pip install -r requirements.txt
python model.py


Make sure the exams.csv file is placed in the same directory.

🎯 Conclusion

Unsupervised clustering revealed hidden grouping patterns in student performance.

Adding these cluster insights into classifiers improved prediction accuracy.

Useful for educational analytics and early academic intervention.

📌 Future Enhancements

Hyperparameter tuning of SVM & Logistic Regression

Use advanced clustering (DBSCAN, OPTICS)

Deploy as a web dashboard (Streamlit/Flask)

Handle class imbalance if present

✍️ Author

Aditya Nair
B.Tech CSE (AI & ML)
Machine Learning / Frontend Development
