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
image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum / image_sum.max()
plt.imshow(image_bw, cmap='gray')
plt.show()
print(image_bw.max())

X = image_bw.reshape(image_bw.shape[0], -1)
plt.hist(X.ravel(), bins=256, color='gray')
plt.show()

pca = PCA()
pca.fit(X)

variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

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
