import tensorflow as tf
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten, Lambda
from keras.optimizers import Adam

def construct_model(shape):
    print(shape)
    model = Sequential()
    model.add(Dense(100, input_dim=shape[1]))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(1048, activation='relu'))
    model.add(Dense(shape[1]-1, activation='sigmoid')) #sigmoid / binary , vs softmax, categorical
    opt = Adam(lr=0.001)
    model.compile(opt, loss='binary_crossentropy', metrics=['acc'])
    return model
    
def gen_training_data(nsamples, digits):
    import random
    #xtrain = np.random.randint(low=0, high=2, size=(nsamples, 100))
    xtrain = np.zeros((nsamples, digits), dtype=int)
    ytrain = np.zeros((nsamples, 1), dtype=int)
    
    for i in range(nsamples):
        goal = random.randrange(0,digits)
        while goal > 0:
            select = random.randrange(0,digits)
            if xtrain[i][select] == 0:
                goal = goal - 1
                xtrain[i][select] =1
        
        #xtrain[i] = np.random.uniform(low=0, high=1.0, size=xtrain[0].shape)
        ytrain[i] = sum(xtrain[i])
        
    return xtrain, ytrain


def main():
    #Sample with math function sum(y) = x
    #[z, x] y
    digits = 20
    xtrain, ytrain = gen_training_data(1000, digits)
    xtest, ytest = gen_training_data(10, digits)
    #cfuzz = np.zeros(xtrain.shape, dtype=int)
    cfuzz = np.random.randint(low=0, high=2, size=xtrain.shape, dtype=int)
    
    print(cfuzz.shape)
    print(xtrain.shape)
    print(ytrain.shape)
    ctrain = np.concatenate((cfuzz, ytrain), axis=1)#np.array([cfuzz + ytrain])
    print(ctrain.shape)
    print(ctrain[0])
    
    model = construct_model(ctrain.shape)
    model.fit(x=ctrain, y=xtrain, epochs=200, shuffle=True, batch_size=100)
    preds = model.predict(np.concatenate((xtest, ytest), axis=1))#np.array[xtest, ytest])
    for i in range(xtest.shape[0]):
        #print(preds
        print("Actual: ", sum(xtest[i]), " Generated: ", round(sum(preds[i])))
    print(preds[-1])




main()