import os

from A4_datawrangling import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_elbow_method(X, standardize=False, filepath="clustering/elbow_plot.png"):
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    inertia_scores = []
    kmeans_params = range(1, 10)
    for k in kmeans_params:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertia_scores.append(kmeans.inertia_)

    # perform Elbow Method analysis
    plt.figure(figsize=(16, 8))
    plt.plot(kmeans_params, inertia_scores, marker='x', linestyle='-', color='b')
    plt.title(f"Elbow Method {'(Standardized)' if standardize else '(Raw)'}")
    plt.xlabel('k')
    plt.ylabel('Inertia scores')
    plt.tight_layout()

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    
    return inertia_scores

def run_kmeans_with_pca(X, y, k=4, standardize=False, filepath="clustering/kmeans_pca.png"):
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(X_pca)

    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df["predicted"] = clusters
    df["target"] = y
    
    unique_labels = np.unique(y)
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    actual_colors = [label_to_color[label] for label in y]

    # compare actual cluster versus KMeans predicted cluster
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].set_title("Actual", fontsize=18)
    ax[0].scatter(df['PC1'], df['PC2'], c=actual_colors, cmap=plt.cm.Set1)
    ax[1].set_title(f"KMeans (k={k})", fontsize=18)
    ax[1].scatter(df['PC1'], df['PC2'], c=df["predicted"], cmap=plt.cm.Set1)

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":
    run_elbow_method(feature_columns, standardize=False, filepath="output/clustering/elbow_raw.png")
    run_elbow_method(feature_columns, standardize=True, filepath="output/clustering/elbow_standardized.png")
    run_kmeans_with_pca(feature_columns, target_column, k=4, standardize=False, filepath="output/clustering/kmeans_pca_raw.png")
    run_kmeans_with_pca(feature_columns, target_column, k=4, standardize=True, filepath="output/clustering/kmeans_pca_standardized.png")