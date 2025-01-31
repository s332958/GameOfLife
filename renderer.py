import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    Esempio:
    0 -> bianco, 1 -> rosso, 2 -> blu, ecc.
    """
    # Se l'ID è negativo o superiore a 63, assegniamo un valore predefinito (nero)
    if id < 0 or id > 63:
        id = 64

    colormap = {
        0: (255, 255, 255),        # Bianco
        1: (255, 0, 0),            # Rosso
        2: (0, 0, 255),            # Blu
        3: (0, 255, 0),            # Verde
        4: (255, 255, 0),          # Giallo
        5: (255, 165, 0),          # Arancione
        6: (128, 0, 128),          # Viola
        7: (255, 192, 203),        # Rosa
        8: (0, 255, 255),          # Ciano
        9: (255, 105, 180),        # Rosa chiaro
        10: (128, 128, 0),         # Oliva
        11: (0, 128, 128),         # Teal
        12: (0, 0, 128),           # Blu scuro
        13: (255, 69, 0),          # Rosso arancio
        14: (186, 85, 211),        # Orchidea scuro
        15: (255, 20, 147),        # Deep pink
        16: (0, 255, 127),         # Verde menta
        17: (255, 228, 196),       # Bisque
        18: (75, 0, 130),          # Indigo
        19: (34, 139, 34),         # Verde foresta
        20: (255, 0, 255),         # Magenta
        21: (128, 128, 128),       # Grigio
        22: (0, 0, 205),           # Blu marino
        23: (0, 255, 0),           # Lime
        24: (255, 240, 245),       # Lavanda
        25: (0, 255, 64),          # Verde brillante
        26: (255, 69, 255),        # Rosso magenta
        27: (240, 128, 128),       # Salmone
        28: (60, 179, 113),        # Verde di mare
        29: (255, 255, 224),       # Giallo chiaro
        30: (205, 92, 92),         # Rosso scuro
        31: (255, 215, 0),         # Giallo dorato
        32: (0, 100, 0),           # Verde scuro
        33: (144, 238, 144),       # Verde chiaro
        34: (123, 104, 238),       # Blu mediano
        35: (240, 230, 140),       # Giallo oliva chiaro
        36: (220, 20, 60),         # Crimson
        37: (255, 99, 71),         # Tomato
        38: (255, 140, 0),         # Arancio scuro
        39: (255, 228, 181),       # Mocaccino
        40: (105, 105, 105),       # Grigio scuro
        41: (0, 128, 0),           # Verde
        42: (255, 240, 245),       # Lavanda
        43: (70, 130, 180),        # Acciaio blu
        44: (255, 182, 193),       # Rosa scuro
        45: (255, 160, 122),       # Arancio chiaro
        46: (255, 255, 0),         # Giallo
        47: (255, 20, 147),        # Rosa profondo
        48: (238, 130, 238),       # Orchidea
        49: (255, 99, 71),         # Tomato
        50: (240, 128, 128),       # Salmone
        51: (143, 188, 143),       # Grigio verde chiaro
        52: (255, 228, 196),       # Bisque
        53: (0, 139, 139),         # Verde mare scuro
        54: (144, 238, 144),       # Verde chiaro
        55: (135, 206, 250),       # Blu chiaro
        56: (255, 165, 0),         # Arancione
        57: (139, 0, 139),         # Viola scuro
        58: (123, 104, 238),       # Blu mediato
        59: (0, 0, 255),           # Blu
        60: (255, 99, 71),         # Tomato
        61: (128, 0, 0),           # Rosso scuro
        62: (255, 140, 0),         # Arancio
        63: (0, 0, 139),           # Blu marino scuro
        64: (255, 255, 255)        # Bianco (per ostacoli)
    }
    return colormap.get(id, (255, 255, 255))  # Restituisce bianco come colore di fallback

# Funzione per aggiornare l'animazione
def aggiorna(frame, matrici1, matrici2, img, ax):
    """
    Funzione di aggiornamento per animare le matrici.
    Mostra una matrice alla volta.
    """
    matrice_corrente_1 = matrici1[frame]  # Nitidezza
    matrice_corrente_2 = matrici2[frame]  # ID per il colore

    # Crea un'immagine a colori utilizzando operazioni vettoriali
    immagine_colori = np.zeros((matrice_corrente_1.shape[0], matrice_corrente_1.shape[1], 3), dtype=np.uint8)

    # Crea una mappa di colori per gli ID
    colori = np.array([mappa_colori(id) for id in range(65)])  # 64 possibili ID per i colori

    # Assicurati che i valori di matrice_corrente_2 siano compresi tra 0 e 64
    matrice_corrente_2 = np.clip(matrice_corrente_2, 0, 64)

    colori_id = colori[matrice_corrente_2] 

    # Aggiungi la nitidezza (luminosità) alla saturazione del colore
    immagine_colori = (colori_id * (matrice_corrente_1[:, :, np.newaxis] / 255)).astype(np.uint8)

    img.set_data(immagine_colori)  # Usa set_data per aggiornare solo l'immagine
    ax.set_title(f"Matrice {frame + 1}")  # Aggiorna il titolo
    return img, ax

if __name__ == "__main__":

    file_path1 = "data/output/mondo0.txt"  # Percorso del file della matrice 1 (nitidezza)
    file_path2 = "data/output/id_matrix0.txt"  # Percorso del file della matrice 2 (ID)

    # Leggi entrambe le matrici
    matrici1 = leggi_matrici(file_path1)
    matrici2 = leggi_matrici(file_path2)
    
    # Imposta la figura e l'asse
    fig, ax = plt.subplots()
    img = ax.imshow(matrici1[0], cmap='gray', interpolation='nearest')
    ax.set_title("Matrice 1")
    ax.axis('off')  # Rimuove gli assi

    # Crea l'animazione
    anim = FuncAnimation(
        fig, 
        aggiorna, 
        fargs=(matrici1, matrici2, img, ax),
        frames=len(matrici1), 
        interval=20,  # Durata di ogni frame in millisecondi
        blit=True
    )

    # Mostra l'animazione
    plt.show()
