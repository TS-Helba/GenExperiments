import tensorflow as tf
import keras
import numpy as np
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten, Lambda
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def construct_model(shape, nclasses):
    #print(shape)
    model = Sequential()
    model.add(Dense(100, input_dim=shape[1]))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(256, activation='relu'))
    #model.add(Dense(1048, activation='relu'))
    #model.add(Dense(1048, activation='relu'))
    model.add(Dense(shape[1]-nclasses, activation='sigmoid')) #sigmoid / binary , vs softmax, categorical
    opt = Adam(lr=0.001)
    model.compile(opt, loss='binary_crossentropy', metrics=['acc'])
    return model
    
def onehotnp(ydata, nclasses):
    newy = np.zeros((ydata.shape[0], nclasses))
    for i in range(ydata.shape[0]):
        newy[i][ydata[i]] = 1
    return newy

    
def gen_training_data():
    import random
    #xtrain = np.random.randint(low=0, high=2, size=(nsamples, 100))
    #xtrain = np.zeros((nsamples, digits), dtype=int)
    #ytrain = np.zeros((nsamples, 1), dtype=int)
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    ytrain = onehotnp(ytrain, 10)
    ytest = onehotnp(ytest, 10)
    
    xtrain = xtrain.reshape((xtrain.shape[0], 784))
    xtest = xtest.reshape((xtest.shape[0], 784))
    
    return xtrain.astype(np.float32) / 255.0, ytrain, xtest.astype(np.float32) / 255.0, ytest


def main():
    #Sample with math function sum(y) = x
    #[z, x] y
    xtrain, ytrain, xtest, ytest = gen_training_data()
    #if False:
    #    xtrain = xtrain[:500]
    #    ytrain = ytrain[:500]
        

    #cfuzz = np.zeros(xtrain.shape, dtype=int)
    cfuzz = np.random.uniform(low=0, high=1, size=xtrain.shape)
    
    #print(cfuzz.shape)
    #print(xtrain.shape)
    #print(ytrain.shape)
    ctrain = np.concatenate((cfuzz, ytrain), axis=1)#np.array([cfuzz + ytrain])
    #print(ctrain.shape)
    #print(ctrain[0])
    
    model = construct_model(ctrain.shape, 10)
    model.fit(x=ctrain, y=xtrain, epochs=25, shuffle=True, batch_size=100)
    preds = model.predict(np.concatenate((xtest, ytest), axis=1))#np.array[xtest, ytest])
    for i in range(10):
        #print(preds
        #print("Actual: ", sum(xtest[i]), " Generated: ", round(sum(preds[i])))
        #plt.image.imsave(str(i)+'img.png', preds[i])
        plt.imshow(preds[i].reshape((28,28)), cmap='gray')
        plt.savefig(str(np.argmax(ytest[i]))+'img'+str(i)+'.png')
    #print(preds[-1])

main()

