import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

dataset = "Tree_growth.csv"

data = pd.read_csv(dataset)

data.dropna(axis = 0)

# Determinamos el cambio teniendo en cuenta el valor anterior
data["TL Change"] = data["TL"].diff()

# Funcion para elaborar un grafico de la valoracion y la valoracion una vez hecho el cambio
def grafico1():
    plt.figure(figsize = (15, 10))
    plt.subplot(2,1,1)
    plt.plot(data["Review Date"], data["Rating"])
    plt.xlabel("Fecha")
    plt.ylabel("Valoracion")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(data["Review Date"], data["Rating Change"])
    plt.xlabel("Fecha")
    plt.ylabel("Valoracion cambiada")
    plt.grid(True)
    plt.show()

grafico1()


# Use the daily change in gold price as the observed measurements X.
X = data[["Rating Change"]].values
# Build the HMM model and fit to the gold price change data.
model = hmm.GaussianHMM(n_components = 3, covariance_type = "diag", n_iter = 50, random_state = 42)
model.fit(X)
# Predict the hidden states corresponding to observed X.
Z = model.predict(X)
states = pd.unique(Z)


def grafico2():
    plt.figure(figsize = (15, 10))
    plt.subplot(2,1,1)
    for i in states:
        want = (Z == i)
        x = data["Review Date"].iloc[want]
        y = data["Rating"].iloc[want]
        plt.plot(x, y, '.')
    plt.legend(states, fontsize=16)
    plt.grid(True)
    plt.xlabel("Review Date", fontsize=16)
    plt.ylabel("Rating", fontsize=16)
    plt.subplot(2,1,2)
    for i in states:
        want = (Z == i)
        x = data["Review Date"].iloc[want]
        y = data["Rating Change"].iloc[want]
        plt.plot(x, y, '.')
    plt.legend(states, fontsize=16)
    plt.grid(True)
    plt.xlabel("Review Date", fontsize=16)
    plt.ylabel("Rating Change", fontsize=16)
    plt.show()

grafico2()