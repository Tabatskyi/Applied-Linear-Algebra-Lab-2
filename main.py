import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA


def find_eigens(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    for i in range(len(eigenvalues)):
        Av = np.dot(matrix, eigenvectors[:, i])
        lv = eigenvalues[i] * eigenvectors[:, i]
        if not np.allclose(Av, lv):
            print(f"A*v != λ*v for eigenvalue λ={eigenvalues[i]} and eigenvector v={eigenvectors[:, i]}")
    return eigenvalues, eigenvectors


print(find_eigens(np.array([[1, 2], [2, 1]])))

image_raw = imread("image2.jpg")
image_bw = np.mean(image_raw, axis=2) / 255

X = image_bw.reshape(-1, 1)

pca = PCA()
pca.fit(X)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

n_components = np.where(cumulative_variance >= 0.95)[0][0] + 1

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axvline(x=n_components, color='r', linestyle='--')
plt.title('Cumulative Variance Explained by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance')
plt.legend()
plt.show()

print(f'Number of components needed to explain 95% of the variance: {n_components}')