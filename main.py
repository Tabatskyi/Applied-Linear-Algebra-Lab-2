import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def find_eigens(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    for i in range(len(eigenvalues)):
        Av = np.dot(matrix, eigenvectors[:, i])
        lv = eigenvalues[i] * eigenvectors[:, i]
        if not np.allclose(Av, lv):
            print(f"A*v != λ*v for eigenvalue λ={eigenvalues[i]} and eigenvector v={eigenvectors[:, i]}")
    return eigenvalues, eigenvectors


print(find_eigens(np.array([[1, 2], [2, 1]])))
