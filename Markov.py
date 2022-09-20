import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

dataset = "chocolate_ratings.csv"

data = pd.read_csv(dataset)

# Transformamos la fecha de una cadena de texto a un datetime object
data["Review Date"] = pd.to_datetime(data["Review Date"])

# Determinamos el cambio en la valoracion
data["Rating change"] = data["Rating"].diff()

# Funcion para elaborar un grafico de la valoracion y la valoracion una vez hecho el cambio
def grafico1():
    plt.figure(figsize = (15, 10))
    plt.subplot(2,1,1)
    plt.plot(data["datetime"], data["gold_price_usd"])
    plt.xlabel("datetime")
    plt.ylabel("gold price (usd)")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(data["datetime"], data["gold_price_change"])
    plt.xlabel("datetime")
    plt.ylabel("gold price change (usd)")
    plt.grid(True)
    plt.show()

grafico1()


# Use the daily change in gold price as the observed measurements X.
X = data[["gold_price_change"]].values
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
        x = data["datetime"].iloc[want]
        y = data["gold_price_usd"].iloc[want]
        plt.plot(x, y, '.')
    plt.legend(states, fontsize=16)
    plt.grid(True)
    plt.xlabel("datetime", fontsize=16)
    plt.ylabel("gold price (usd)", fontsize=16)
    plt.subplot(2,1,2)
    for i in states:
        want = (Z == i)
        x = data["datetime"].iloc[want]
        y = data["gold_price_change"].iloc[want]
        plt.plot(x, y, '.')
    plt.legend(states, fontsize=16)
    plt.grid(True)
    plt.xlabel("datetime", fontsize=16)
    plt.ylabel("gold price change (usd)", fontsize=16)
    plt.show()

grafico2()