import os
import numpy as np
from PIL import Image

def normalize_image(image_array):
    """
    Normalizza un array immagine affinché la somma dei valori sia pari a 1.
    """
    total = np.sum(image_array)
    if total == 0:
        return image_array  # Evita divisione per zero
    return image_array / total

def process_images(input_folder="./data/custom_kernels", output_folder="./filters/custom_filters"):
    """
    Legge immagini grayscale .png 8 bit da una cartella e salva matrici normalizzate in file .txt.

    :param input_folder: Cartella da cui leggere le immagini.
    :param output_folder: Cartella in cui salvare i file normalizzati.
    """
    if not os.path.exists(input_folder):
        print(f"Errore: La cartella '{input_folder}' non esiste.")
        return

    # Creare la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Itera sui file nella cartella di input
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)

            # Carica l'immagine e la converte in array
            with Image.open(input_path) as img:
                img = img.convert("L")  # Assicura che sia grayscale
                img_array = np.array(img, dtype=np.float32)

            # Normalizza la matrice
            normalized_array = normalize_image(img_array)

            # Ottieni le dimensioni dell'immagine
            height, width = normalized_array.shape

            # Prepara il nome del file di output
            output_filename = f"Cfilter_{os.path.splitext(filename)[0]}.txt"
            output_path = os.path.join(output_folder, output_filename)

            # Scrive la dimensione e la matrice normalizzata nel file
            with open(output_path, "w") as f:
                f.write(f"{height}\n")  # Scrive la dimensione
                for row in normalized_array:
                    f.write(" ".join(f"{value:.6f}" for value in row) + "\n")

    print(f"Elaborazione completata. File salvati in '{output_folder}'.")

# Esegui il programma
process_images()
