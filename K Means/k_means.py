import numpy as np
import matplotlib.pyplot as plt

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)
class KMeans:
    def __init__(self, K=5, max_iters = 100, plot_steps = False):
        self.K = K
        self.max_iters =max_iters
        self.plot_steps =plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape            
        
        # Initialize centroids randomly
        random_sample = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample]

        #optimize clusters
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            

            if self._is_converged(centroid_old, self.centroids): 
                break

            if self.plot_steps:
                self.plot()
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_ids, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_ids[sample_idx]
            
            return labels

    def _create_clusters(self, centroids):
        clusters =[[] for _ in range(self.K)]        
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids) 
            clusters[centroid_idx].append(idx)
        return clusters
    def _closest_centroid(self, sample, centroids):
         distances = [euclidean_dist(sample, point) for point in centroids]
         closest_idx = np.argmin(distances)
         return closest_idx

    def _get_centroids(self, clusters):
        #assign mean value of cluster to centroid
        centroids = np.zeros((self.K, self.n_features))
        for cluster_ids, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis =0)
            centroids[cluster_ids] = cluster_mean
        return centroids
        
    def _is_converged(self, old_centroids, new_centroids):
        #distances between old and new centroids
        distances = [euclidean_dist(old_centroids[i], new_centroids[i]) for i in range(self.K)]

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="*", color="black", linewidth =2)
        plt.show()
# Testing
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()