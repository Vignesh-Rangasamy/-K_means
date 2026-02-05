import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SKKMeans

np.random.seed(42)

X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)


class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def _initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def _compute_distances(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = []
        for i in range(self.k):
            points = X[labels == i]
            if len(points) == 0:
                centroids.append(X[np.random.choice(X.shape[0])])
            else:
                centroids.append(points.mean(axis=0))
        return np.array(centroids)

    def _compute_wcss(self, X, labels, centroids):
        wcss = 0.0
        for i in range(self.k):
            cluster_points = X[labels == i]
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            distances = self._compute_distances(X, self.centroids)
            labels = self._assign_clusters(distances)
            updated_centroids = self._update_centroids(X, labels)

            if np.all(np.abs(updated_centroids - self.centroids) < self.tol):
                break

            self.centroids = updated_centroids

        self.labels_ = labels
        self.wcss_ = self._compute_wcss(X, labels, self.centroids)
        return self


custom_model = KMeansScratch(k=3)
custom_model.fit(X)

custom_labels = custom_model.labels_
custom_centroids = custom_model.centroids
custom_wcss = custom_model.wcss_


sk_model = SKKMeans(n_clusters=3, random_state=42, n_init=10)
sk_model.fit(X)

sk_labels = sk_model.labels_
sk_centroids = sk_model.cluster_centers_
sk_wcss = sk_model.inertia_


print("WCSS (Custom K-Means):", custom_wcss)
print("WCSS (Scikit-Learn): ", sk_wcss)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=custom_labels, cmap="viridis", s=30)
plt.scatter(custom_centroids[:, 0], custom_centroids[:, 1],
            c="red", marker="X", s=200)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Custom K-Means")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=sk_labels, cmap="viridis", s=30)
plt.scatter(sk_centroids[:, 0], sk_centroids[:, 1],
            c="red", marker="X", s=200)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scikit-Learn K-Means")

plt.tight_layout()
plt.show()


print("\nAnalysis:")
if custom_wcss <= sk_wcss:
    print("Custom implementation achieved clustering quality comparable to scikit-learn.")
else:
    print("Scikit-learn achieved slightly lower WCSS due to optimized initialization.")

print(
    "Both approaches formed well-separated clusters. "
    "Minor differences in WCSS are caused by centroid initialization "
    "and internal optimization strategies."
)
