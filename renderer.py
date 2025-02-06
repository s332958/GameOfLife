import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse  # Importa argparse per gestire gli argomenti da riga di comando

# Funzione per caricare la matrice dal file
def load_image_from_txt(file_path):
    return np.loadtxt(file_path, dtype=np.uint8)

def leggi_matrici(file_path):
    """
    Legge un file di testo contenente matrici e le restituisce come un array di matrici numpy.
    """
    matrici = []
    with open(file_path, 'r') as f:
        contenuto = f.read()
        blocchi = contenuto.strip().split('\n\n')  # Divide le matrici separate da una riga vuota
        for blocco in blocchi:
            matrice = np.loadtxt(blocco.splitlines(), dtype=np.uint8)
            matrici.append(matrice)
    return np.array(matrici)

# Funzione per ottenere il colore in base all'ID
def mappa_colori(id):
    """
    Restituisce un colore associato all'ID.
    """
    if id < 0 or id > 63:
        id = 64

    colormap = {
        0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0),
        4: (255, 255, 0), 5: (255, 165, 0), 6: (128, 0, 128), 7: (255, 192, 203),
        8: (0, 255, 255), 9: (255, 105, 180), 10: (128, 128, 0), 11: (0, 128, 128),
        12: (0, 0, 128), 13: (255, 69, 0), 14: (186, 85, 211), 15: (255, 20, 147),
        16: (0, 255, 127), 17: (255, 228, 196), 18: (75, 0, 130), 19: (34, 139, 34),
        20: (255, 0, 255), 21: (128, 128, 128), 22: (0, 0, 205), 23: (0, 255, 0),
        24: (255, 240, 245), 25: (0, 255, 64), 26: (255, 69, 255), 27: (240, 128, 128),
        28: (60, 179, 113), 29: (255, 255, 224), 30: (205, 92, 92), 31: (255, 215, 0),
        32: (0, 100, 0), 33: (144, 238, 144), 34: (123, 104, 238), 35: (240, 230, 140),
        36: (220, 20, 60), 37: (255, 99, 71), 38: (255, 140, 0), 39: (255, 228, 181),
        40: (105, 105, 105), 41: (0, 128, 0), 42: (255, 240, 245), 43: (70, 130, 180),
        44: (255, 182, 193), 45: (255, 160, 122), 46: (255, 255, 0), 47: (255, 20, 147),
        48: (238, 130, 238), 49: (255, 99, 71), 50: (240, 128, 128), 51: (143, 188, 143),
        52: (255, 228, 196), 53: (0, 139, 139), 54: (144, 238, 144), 55: (135, 206, 250),
        56: (255, 165, 0), 57: (139, 0, 139), 58: (123, 104, 238), 59: (0, 0, 255),
        60: (255, 99, 71), 61: (128, 0, 0), 62: (255, 140, 0), 63: (0, 0, 139),
        64: (255, 255, 255)  # Bianco per ostacoli
    }
    return colormap.get(id, (255, 255, 255))

# Funzione per aggiornare l'animazione
def aggiorna(frame, matrici1, matrici2, img, ax):
    """
    Funzione di aggiornamento per animare le matrici.
    """
    matrice_corrente_1 = matrici1[frame]
    matrice_corrente_2 = matrici2[frame]

    immagine_colori = np.zeros((matrice_corrente_1.shape[0], matrice_corrente_1.shape[1], 3), dtype=np.uint8)

    colori = np.array([mappa_colori(id) for id in range(65)])

    matrice_corrente_2 = np.clip(matrice_corrente_2, 0, 64)

    colori_id = colori[matrice_corrente_2] 

    immagine_colori = (colori_id * (matrice_corrente_1[:, :, np.newaxis] / 180)).astype(np.uint8)

    img.set_data(immagine_colori)
    ax.set_title(f"Matrice {frame + 1}")
    return img, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza l'animazione delle matrici di un mondo specifico.")
    parser.add_argument("n", type=int, help="Numero del mondo (n) da visualizzare.")

    args = parser.parse_args()
    n = args.n  # Ottiene il valore di n dalla riga di comando

    file_path1 = f"data/output/mondo{n}.txt"
    file_path2 = f"data/output/id_matrix{n}.txt"

    # Leggi le matrici
    matrici1 = leggi_matrici(file_path1)
    matrici2 = leggi_matrici(file_path2)
    
    # Imposta la figura e l'asse
    fig, ax = plt.subplots()
    img = ax.imshow(matrici1[0], cmap='gray', interpolation='nearest')
    ax.set_title(f"Matrice {n}")
    ax.axis('off')

    # Crea l'animazione
    anim = FuncAnimation(
        fig, 
        aggiorna, 
        fargs=(matrici1, matrici2, img, ax),
        frames=len(matrici1), 
        interval=0,  
        blit=True
    )

    # Mostra l'animazione
    plt.show()
