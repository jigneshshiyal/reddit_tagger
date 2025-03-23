from sklearn.cluster import KMeans
import umap

def get_clusters(embeddings, num_clusters=3):
    """
    Perform clustering on the given embeddings using UMAP and KMeans.

    Args:
        embeddings (list): List of embeddings to cluster.
        num_clusters (int): Number of clusters to form.

    Returns:
        list: Cluster labels for each embedding.
    """
    # Reduce dimensionality with UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = umap_reducer.fit_transform(embeddings)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_2d)

    return clusters, embeddings_2d