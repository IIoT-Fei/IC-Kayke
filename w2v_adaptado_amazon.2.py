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
        
        w = filter(lambda k: k != '', w)
        
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


words = stop_words[ int(len(stop_words)*0.2):]
idx = dict([(b,a) for (a,b) in enumerate(words)])

def gera_treinamento(janelas):
    for j in janelas:
        m = int(len(j)/2)
        yield (j[m], [*j[0:m], *j[m+1:]])

def cria_vetores_treinamento():
    x_train = []
    y_train = []
    for (a,b) in tqdm(gera_treinamento(win(filter(lambda x: x in words, palavras()),2))):
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

def embbeded(word):
    x = [0] * len(words)
    for i in words:
            x[idx[i]] = 1
    return x


def carrega_vetores_treinamento():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    return x_train, y_train

#print("Criando vetores Treinamento")
#cria_vetores_treinamento()
(x_train, y_train) = carrega_vetores_treinamento()
#print("Vetores de Treinamento carregados")

sz_words = x_train.shape[1]

entrada = tf.placeholder(tf.float32,[1,sz_words])
saida_esperada = tf.placeholder(tf.float32,[1,sz_words])

w2v = tf.Variable(tf.random_normal([sz_words, 2]))
b_w2v = tf.Variable(tf.random_normal([1]))

w_saida = tf.Variable(tf.random_normal([2, sz_words]))
b_saida = tf.Variable(tf.random_normal([1]))

c1 = tf.add(tf.matmul(entrada, w2v), b_w2v)
c2 = tf.add(tf.matmul(c1, w_saida), b_saida)


loss = tf.losses.mean_pairwise_squared_error(saida_esperada,c2)
ls = tf.losses.softmax_cross_entropy(saida_esperada,c2)
otm = tf.train.AdamOptimizer(0.005).minimize(loss)

s = tf.Session()
s.run(tf.global_variables_initializer())
for i in range(1):
    stat=tqdm(zip(x_train, y_train))
    for (a,b) in stat:
        l,_ = s.run([loss, otm], {entrada: [a], saida_esperada: [b]})
        stat.set_description("erro: %04.2f" % l)
    
    print("%d vez: erro: %f" % (i,l))


def testa(idx):
    x = [0]*len(words)
    x[idx]=1
    print(words[idx])
    x = np.array([x])

    saida = s.run([c2], {entrada: [x]})
    for ww in list(reversed(np.argsort(saida)))[0][:5]:
        print(words[ww])
    print("finalizei")


testa(2)