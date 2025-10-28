import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# NEW imports for cluster validation metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import dendrogram utilities
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("exams.csv")

# Encode categorical
gender_map = {"male": 0, "female": 1}
race_map = {"group a": 0, "group b": 1, "group c": 2, "group d": 3, "group e": 4}
parent_edu_map = {
    "some high school": 0, "high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
}
lunch_map = {"standard": 0, "free/reduced": 1}
prep_course_map = {"none": 0, "completed": 1}

for col, mapping in [
    ('gender', gender_map),
    ('race/ethnicity', race_map),
    ('parental level of education', parent_edu_map),
    ('lunch', lunch_map),
    ('test preparation course', prep_course_map)
]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map(mapping)

# Drop NA rows
df = df.dropna().copy()

# Target
df['pass_math'] = (df['math score'] >= 50).astype(int)

# Features
X = df[['gender', 'race/ethnicity', 'parental level of education',
        'lunch', 'test preparation course', 'reading score', 'writing score']]
y = df['pass_math']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================
# Dendrogram
# =======================
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 7))
dendrogram(
    linked,
    truncate_mode='level',  # shows upper hierarchy clearly
    p=5,
    leaf_rotation=90.,
    leaf_font_size=10.
)
plt.title("Hierarchical Clustering Dendrogram (Ward Method)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

print("\nDendrogram generated successfully ✅")

# =======================
# Clustering Algorithms
# =======================
n_clusters = 3

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
df['gmm_cluster'] = gmm.fit_predict(X_scaled)

# Agglomerative Clustering
hier = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['hier_cluster'] = hier.fit_predict(X_scaled)

print("\nClustering Completed ✅")

# =======================
# Cluster Validation Metrics
# =======================
def safe_cluster_metrics(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None, None, None
    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels)
    )

cluster_metrics = {}
for method, labels in [
    ("KMeans", df['kmeans_cluster'].values),
    ("GMM", df['gmm_cluster'].values),
    ("Hierarchical", df['hier_cluster'].values)
]:
    cluster_metrics[method] = dict(zip(
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        safe_cluster_metrics(X_scaled, labels)
    ))

print("\n=== Cluster Quality Metrics ===")
for method, metrics in cluster_metrics.items():
    print(f"\n{method}:")
    if metrics["silhouette"] is None:
        print("  Metrics not computable")
    else:
        print(f"  Silhouette Score         : {metrics['silhouette']:.4f}")
        print(f"  Davies-Bouldin Score     : {metrics['davies_bouldin']:.4f}")
        print(f"  Calinski-Harabasz Score  : {metrics['calinski_harabasz']:.4f}")

# =======================
# Model Training with Cluster Features
# =======================
X_enriched = df[['gender', 'race/ethnicity', 'parental level of education',
                 'lunch', 'test preparation course', 'reading score', 'writing score',
                 'kmeans_cluster', 'gmm_cluster', 'hier_cluster']]

X_train, X_test, y_train, y_test = train_test_split(
    X_enriched, y, test_size=0.2, random_state=42, stratify=y
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Bayesian Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, preds),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds),
        roc_auc_score(y_test, probs),
        confusion_matrix(y_test, preds)
    ])

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Confusion Matrix"
]).set_index("Model")

print("\n=== Model Performance ===")
print(results_df.round(4))

# PCA Visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=df['kmeans_cluster'], cmap='viridis', s=40)
plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()

# Saving models
joblib.dump(models["Bayesian Logistic Regression"], "bayes_logit_with_clusters.joblib")
joblib.dump(models["SVM (RBF)"], "svm_with_clusters.joblib")
joblib.dump(scaler, "scaler_with_clusters.joblib")

print("\n Models + Clustering + Dendrogram Completed Successfully!")
