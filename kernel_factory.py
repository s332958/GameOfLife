import numpy as np
import cv2 as cv

size = 15

# Parametro della dimensione del kernel
def RingKernel():
    R = size / 4  # Raggio proporzionale alla dimensione
    width = R / 2  # Larghezza proporzionale a R

    # Creazione della griglia di coordinate
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    distance = np.sqrt((x+1)**2 + (y+1)**2)  # Distanza radiale dal centro

    # Funzione per il kernel ad anello smooth
    ring_kernel = np.exp(-((distance - R)**2) / (2 * width**2))

    # Impostare valori molto piccoli (vicini allo 0) al centro e ai bordi
    ring_kernel[distance < (1)] = 0  # Valori interni
    ring_kernel[distance > (size//2)] = 0  # Valori esterni

    # Normalizzazione del kernel
    ring_kernel /= np.sum(ring_kernel)
    return ring_kernel

def GaussianKernel ():
    R = size / 4  # Raggio proporzionale alla dimensione
    width = R / 2  # Larghezza proporzionale a R

    # Creazione della griglia di coordinate
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    distance = np.sqrt((x+1)**2 + (y+1)**2)  # Distanza radiale dal centro

    # Funzione per il kernel ad anello smooth
    gaussian_kernel = np.exp(-((distance - R)**2) / (2 * width**2))
    gaussian_kernel[distance > (size//2)] = 0 
    gaussian_kernel[distance < R] = 1
    gaussian_kernel[distance < (1)] = 0
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel


kernel = RingKernel()
K_normalized = cv.normalize(kernel, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

with open('data/filters/filter.txt', 'w') as f:
    f.write(f"{size}\n")  # Scrive il valore di size e va a capo
    np.savetxt(f, kernel, fmt='%f', delimiter=' ')  # Scrive la matrice
#cv.imshow("Kernel", K_normalized)
#cv.waitKey(0)
#cv.destroyAllWindows()

