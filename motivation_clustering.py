from llm_pipeline import _sentence_encoder
from sentence_transformers import util as st_util
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_motivation_embeddings(motivations: list[str], bar: bool = True):
    model = _sentence_encoder()        
    embeddings = model.encode(motivations, convert_to_tensor=True, show_progress_bar=bar)
    return embeddings

def cluster_motivations(embeddings, threshold: float = 0.75, min_samples: int = 2, bar: bool = True):
    clusters = st_util.community_detection(embeddings, min_community_size=min_samples, threshold=threshold, show_progress_bar=bar)
    labels = [-1] * len(embeddings)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    return clusters, labels

def embeddings_PCA(embeddings, n_components: int = 2):
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    return pca_embeddings

def embeddings_TSNE(embeddings, n_components: int = 2, perplexity: int = 30, max_iter: int = 1000):    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, random_state=42, metric='cosine')
    tsne_embeddings = tsne.fit_transform(embeddings.cpu().numpy())
    return tsne_embeddings


def plot_clusters_2d(pca_embeddings, labels, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels, cmap='viridis', s=2)    
    plt.title(title)
    plt.show()

def plot_clusters_3d(pca_embeddings, labels, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], pca_embeddings[:, 2], c=labels, cmap='viridis', s=5)
    ax.set_title(title)
    plt.show()
