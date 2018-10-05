import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json
from tqdm import tqdm

with open('teste3.json', 'r') as r:
    base = json.load(r)

textos = [ x["reviewText"] for x in base]
print("tudo lido!")


def palavras():
    for l in textos:
        l=l.lower()
        l=l.replace("\n","")
        l=l.replace("\r","")
        l=l.replace("\t","")
        l=l.replace("("," ")
        l=l.replace(")"," ")
        l=l.replace("."," ")
        l=l.replace(","," ")
        l=l.replace("\x0c","") 
        l=l.replace("  "," ")
        w = l.split(" ")
        
        w = list(filter(lambda k: k != '', w))
        
        for x in w:
            yield x

def win(it, n):
    x = itertools.tee(it,n)
    for i in range(n):
        for i2 in range(i,n):
            next(x[i2])
        
    for z in zip(*x):
        yield z

d = {}
for x in palavras():
    if x in d:
        d[x]+=1
    else:
        d[x] = 1

stop_words = sorted(d.keys(), key=lambda x: d[x], reverse=True)
stop_words_count = [d[x] for x in stop_words]


words = stop_words[ int(len(stop_words)*0.4):]
idx = dict([(b,a) for (a,b) in enumerate(words)])

w2c = tf.layers.Dense(12,kernel_initializer=tf.random_uniform_initializer)
out_layer = tf.layers.Dense(len(words), activation=lambda x: tf.nn.softmax, kernel_initializer=tf.random_uniform_initializer,
bias_initializer=tf.random_uniform_initializer)
model = keras.Sequential([
    w2c,
    out_layer
])

def gera_treinamento(janelas):
    for j in janelas:
        m = int(len(j)/2)
        yield (j[m], [*j[0:m], *j[m+1:]])

def cria_vetores_treinamento():
    x_train = []
    y_train = []
    for (a,b) in tqdm(gera_treinamento(win(filter(lambda x: x in words, palavras()),5))):
        x = [0] * len(words)
        x[idx[a]] = 1
        y_train.append(x)
        x = [0] * len(words)
        for i in b:
            x[idx[i]] = 1
        x_train.append(x)
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    np.save("x_train", x_train)
    np.save("y_train", y_train)

def carrega_vetores_treinamento():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    return x_train, y_train

print("Criando vetores Treinamento")
#cria_vetores_treinamento()
(x_train, y_train) = carrega_vetores_treinamento()
print("Vetores de Treinamento carregados")

print("x=")
print(x_train[1,:])
print("y=")
print(y_train[1,:])

model.compile(tf.train.AdamOptimizer(0.001), tf.losses.softmax_cross_entropy )
model.fit(x_train, y_train, epochs=50)

def testa(idx):
    x = [0]*len(words)
    x[idx]=1
    print(words[idx])
    x = np.array([x])

    saida = model.predict(x)
    for ww in list(reversed(np.argsort(saida)))[0][:5]:
        print(words[ww])
    print("finalizei")


testa(2)