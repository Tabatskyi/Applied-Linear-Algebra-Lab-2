import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA
from multiprocessing.dummy import Pool as ThreadPool


def find_eigens(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    for i in range(len(eigenvalues)):
        Av = np.dot(matrix, eigenvectors[:, i])
        lv = eigenvalues[i] * eigenvectors[:, i]
        if not np.allclose(Av, lv):
            print(f"A*v != λ*v for eigenvalue λ={eigenvalues[i]} and eigenvector v={eigenvectors[:, i]}")
    return eigenvalues, eigenvectors


def plot_pca_variance(components, cumulative_variance):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axvline(x=components, color='r', linestyle='--')
    plt.title('Cumulative Variance Explained by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance')
    plt.legend()
    plt.show()


def pca_reconstruction(image, percentage_variance=0.95, plot=False):
    X = image.reshape(image.shape[0], -1)

    pca = PCA()
    pca.fit(X)

    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)

    components = np.where(cumulative_variance >= percentage_variance)[0][0] + 1

    if plot:
        plot_pca_variance(components, cumulative_variance)

    print(f'Number of components needed to explain {percentage_variance} of the variance: {components}')

    pca = PCA(n_components=components)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)

    return X_reconstructed.reshape(image_bw.shape), components


print(find_eigens(np.array([[1, 2], [2, 1]])))

image_raw = imread('image3.png')
image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum / image_sum.max()
plt.figure(figsize=(20, 15))
plt.imshow(image_bw, cmap='gray')
plt.title('Original Black and White Image', fontsize=20)
plt.tight_layout()
plt.show()
print(image_bw.max())

variance_thresholds = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def run_experiment(variance):
    plot = False
    if variance == 0.95:
        plot = True
    image, components = pca_reconstruction(image_bw, variance, plot)
    return variance, image, components


pool = ThreadPool(6)
results = pool.map(run_experiment, variance_thresholds)
pool.close()
pool.join()

plt.figure(figsize=(20, 15))
for i, (variance_threshold, reconstructed_image, n_components) in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Variance: {variance_threshold * 100}%\nComponents: {n_components}', fontsize=20)
    plt.axis('off')
plt.tight_layout()
plt.show()
