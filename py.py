import numpy as np

# Dimensione della matrice
n = 11

# Creiamo una matrice triangolare superiore di dimensione n x n
matrice = np.triu(np.linspace(0, 1, n**2).reshape(n, n))

# Normalizziamo la matrice in modo che la somma dei suoi valori sia 1
matrice_normalizzata = matrice / matrice.sum()

# Stampa della matrice
print(matrice_normalizzata)
