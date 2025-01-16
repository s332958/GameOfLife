import numpy as np
import cv2 as cv



R = 9

# Genera la griglia di coordinate
y, x = np.ogrid[-R:R, -R:R]  # Genera due array per le coordinate

# Calcola la norma Euclidea (distanza) tra ogni coppia di coordinate
D = np.sqrt(x**2 + y**2) / R  # Norma Euclidea



# Funzione campana (Bell)
def bell(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

K = (D < 1) * bell(D, 0.5, 0.15)
K = K / np.sum(K)

# Normalizza il kernel per visualizzarlo
K_normalized = cv.normalize(K, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Mostra il kernel
np.savetxt('data/kernel_filter.txt.txt', K_normalized, fmt='%d', delimiter=' ')
#cv.imshow("Kernel", K_normalized)
#cv.waitKey(0)
#cv.destroyAllWindows()

