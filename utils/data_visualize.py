import matplotlib.pyplot as plt

def visualize_clusters(embeddings_2d, clusters, num_clusters):
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        plt.scatter(embeddings_2d[clusters == i, 0], embeddings_2d[clusters == i, 1], label=f"Cluster {i}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.title("Reddit Title Clustering")
    plt.show()
