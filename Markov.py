import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

dataset = "Tree_growth.csv"

data = pd.read_csv(dataset)

data.dropna(axis = 0)

# Determinamos el cambio teniendo en cuenta el valor anterior
data["TL Change"] = data["TL"].diff()

# Funcion para elaborar un grafico del dataset aplicando cadenas de markov
def grafico():
    plt.figure(figsize = (15, 10))
    plt.subplot(2,1,1)
    plt.plot(data["N"], data["TL"])
    plt.xlabel("Id")
    plt.ylabel("Longitud total del brote")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(data["N"], data["TL Change"])
    plt.xlabel("Id")
    plt.ylabel("Prediccion Longitud total brote")
    plt.grid(True)
    plt.show()

grafico()