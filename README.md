# Tarea Latex 1

Trabajo realizado por: 

*  Esther Rodríguez 

*  Teresa Álvarez 

*  Rubén Nogueras

*  Pelayo Huerta

*  Jose Luis
  
  # Cadenas de Markov
  
  Una cadena de Markov es un proceso evolutivo que consiste de un número finito de estados en cual la probabilidad de que ocurra un evento depende solamente del evento inmediatamente anterior con unas probabilidades que están fijas, es decir establece una fuerte dependencia entre un evento y otro suceso anterior.
Por este motivo, a menudo se dice que estas cadenas cuentan con memoria.

La base de las cadenas es la conocida como propiedad de Markov, la cual resume lo dicho anteriormente en la siguiente regla: lo que la cadena experimente en un momento t + 1 solamente depende de lo acontecido en el momento t (el inmediatamente anterior).

Dada esta sencilla explicación de la teoría, puede observarse que es posible a través de la misma conocer la probabilidad de que un estado ocurra en el largo plazo. Esto ayuda indudablemente a la predicción y estimación en largos periodos de tiempo.

![image](https://user-images.githubusercontent.com/91721860/193780269-711ade0f-0da7-4dbc-8ec2-340b099418a6.png)


Código realizado:

````
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

dataset = "MarkovChain/Tree_growth.csv"

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
    plt.savefig("MarkovChain/Grafico Cadena Markov.png")

grafico()

````
  
# Dinamic Time Warping (DTW)
Deformación dinámica del tiempo.

Es un algoritmo para medir la similitud entre dos secuencias temporales, que pueden variar las velocidades; es un método que calcula una coincidencia óptima entre dos secuencias dadas con ciertas restricciones y reglas (las cuales satisface).

* Cómo funciona el algoritmo DTW

Se ha aplicado para secuencias temporales de datos de vídeo, audio y gráficos (cualquier dato que pueda convertirse en una secuencia lineal pued analizarse con DTW).
Entre otras, está diseñado para tratar el reconocimiento de voz automático, para afrontar las diferentes velocidades de habla.

Su objetivo es encontrar la alineación global óptima entre dos series de tiempo, explotando las distorsiones temporales entre las dos series de tiempo.

Ejemplo, comparando la distancia euclídea y el DTW

![image](https://user-images.githubusercontent.com/91721860/193786916-10777e35-933b-4cf8-990f-e0f9d57021cd.png)


Código realizado: 

````
import numpy as np


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix


a = [1,2,3]
b = [2,2,2,3,4]

print(dtw(a,b))

````
