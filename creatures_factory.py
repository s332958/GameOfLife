import os
import numpy as np
import random

def generate_creatures_matrices(n_creatures, size, noise_scale=1, apply_circle_mask=False):
    """
    Genera n_creature matrici quadrate di grandezza size con noise su scala controllabile
    e salva i file con il nome creatura_[x].txt nella cartella creatures.
    Ogni file include la dimensione della matrice come prima riga.

    Args:
        n_creatures (int): Numero di creature (matrici) da generare.
        size (int): Dimensione delle matrici quadrate.
        noise_scale (float): Scala dei pattern del noise (più alto = pattern più grandi).
        apply_circle_mask (bool): Se True, imposta a 0 tutti i valori fuori dal cerchio inscritto.
    """
    # Creazione della cartella "creatures" se non esiste
    folder_name = "data/creatures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Generazione e salvataggio delle matrici
    for i in range(n_creatures):
        # Cambia il seed per ogni matrice
        seed = random.randint(0, int(1e6))
        np.random.seed(seed)

        # Funzione di noise basata su una distribuzione casuale controllata
        matrix = np.random.randint(0, 256, (size, size), dtype=int)

        # Applica il cerchio inscritto se richiesto
        if apply_circle_mask:
            center = size // 2
            y, x = np.ogrid[:size, :size]
            distance_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            mask = distance_from_center <= (size / 2)
            matrix[~mask] = 0

        # Nome del file
        file_name = f"creatura_{i+1}.txt"
        file_path = os.path.join(folder_name, file_name)

        # Salva la dimensione e la matrice nel file
        with open(file_path, "w") as file:
            file.write(f"{size}\n")  # Prima riga con la dimensione della matrice
            np.savetxt(file, matrix, fmt="%d")  # Salva la matrice

        print(f"Salvata matrice {i+1} in {file_path} con seed {seed}")

generate_creatures_matrices(n_creatures=16, size=128, noise_scale=256, apply_circle_mask=True)

