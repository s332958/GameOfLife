import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("INIZIO PLOT")
    score = np.loadtxt("log_score.txt")
    plt.plot(score)
    plt.show()