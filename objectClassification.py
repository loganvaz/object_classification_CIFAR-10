#file management

import os
def createFolders(h, n):
    try:
        f = os.mkdir(h+"\\"+n)
        f.close()
    except:
        return h+"\\"+n
    return h+"\\"+n

#loadingModel
loadModel = True
path = createFolders(os.getcwd(), "modelInProgess")+"\\model"


#import _pickle as cPickle
import pickle
print("hello there")

#casting y to one hot data
from keras.utils.np_utils import to_categorical

#for model and np arrays
import numpy as np


#extraction code, from https://www.cs.toronto.edu/~kriz/cifar.html
#Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
def unpickle(file):
    #import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

folder = "cifar-10-batches-py\\"
file = folder + "data_batch_"



##converter I used in previous project
class y_convert:
    listed = list()
    invDict = dict()
    def __init__(self, y_text):
        self.listed = list(set(y_text))
        numClasses = len(self.listed)
        for i in range(numClasses):
            self.invDict[self.listed[i]] = i

        
    def get_one_hot(self, y):
        y = np.array([self.invDict[i] for i in y])
        #toReturn = np.zeros((len(y), len(self.listed)))
        #toReturn[np.arange(len(y), y)] = 1
        return to_categorical(y)
        #toReturn = np.zeros((len(y), len(listed)))
        #for i in range(len(y)):
    def getText(self, y_hat):
        try:#it's prediction
            return [self.listed[i] for i in y_hat]
        except:#it's one_hot
            return [self.listed[np.argmax(i)] for i in y_hat]


#model stuff

#callback to save model based on validation accuracy
from keras.callbacks import Callback,ModelCheckpoint
import tensorflow as tf
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#F1 score, additional metric for analysis
from keras import backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

#at long last, the actual model
from keras.layers import Concatenate, Activation, Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, BatchNormalization, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.keras import regularizers

#new model
def classification_model(inputShape, outputShape):#this is the good model
    #def classification_model(inputShape, outputShape):
    X_input = Input(shape=inputShape)
    print(X_input.shape)

    X = ZeroPadding2D((1,1))(X_input)
    X = Conv2D(64, kernel_size=2, strides=1, padding="same")(X)
    X = BatchNormalization(axis=3,name="zero_batch_norm")(X)
    X = Activation("relu")(X)


    X = MaxPooling2D(pool_size = (2,2))(X)

    print(X.shape)
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(100, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="first_batch_norm")(X)
    X = Activation("relu")(X)
    
    print(X.shape)
    X = Conv2D(128, kernel_size = 4, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="2_batch_norm")(X)
    X = Activation("relu")(X)

    X = MaxPooling2D(pool_size=(2,2))(X)

    print(X.shape)
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(186, kernel_size=3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="3_batch_norm")(X)
    X_ = Activation("relu")(X)

    print(X.shape)
    X = Conv2D(256, kernel_size=4, strides=1, padding="valid")(X_)
    X = BatchNormalization(axis=3, name="4_batch_norm")(X)
    X_0 = Activation("relu")(X)

    print(X.shape)
    X = ZeroPadding2D((1,1))(X_)
    X = Conv2D(256, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis =3, name="5_batch_norm")(X)
    X_1 = Activation("relu")(X)

  

    X = Concatenate()([X, X_1])

    X = Conv2D(512, kernel_size=2, strides=1, padding="same")(X)
    X = BatchNormalization(axis=3, name="5.5_batch_norm")(X)
    X = Activation("relu")(X)
    print("post concat")
    print(X.shape)

    courseX = MaxPooling2D((2,2))(X)
    courseX = Flatten()(courseX)

    courseX_ = Dense(128, activation="relu")(courseX)

    courseX_ = Dropout(0.2)(courseX_)

    courseX = Dense(256, activation="relu")(courseX_)

    courseX = Dropout(0.2)(courseX)

    courseX = Dense(128)(courseX_)

    courseX = Dropout(0.2)(courseX)

    courseX = Activation("relu")(courseX+courseX_)
    #no dropout after this b/c already applied dropout to each

    
    

  
    
    #maybe try dropfilter early on
    X = Conv2D(1024, kernel_size=3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="6_batch_norm")(X)
    X = Activation("relu")(X)
    print(X.shape)
    X = Flatten()(X)

   

    X = Dense(128, activation="relu")(X)

    X = Concatenate()([X, courseX])


    X = Dropout(0.2)(X)

    X = Dense(128, activation="relu")(X)

    X= Dropout(0.2)(X)
    
    X_ = X

    print(X_.shape)
   
    X = Dense(256, activation="relu")(X)

    X = Dropout(.2)(X)
    

    X = Dense(128)(X)

    X = Dropout(0.2)(X)


    print(X.shape)
    print(X_.shape)
    X = Activation("relu")(X+X_)

    #no dropout b/c already applied to each of these
    

    X = Dense(outputShape, activation="softmax")(X)

    model = Model(inputs=X_input, outputs = X)

    return model



#old model
"""
def classification_model(inputShape, outputShape):
    X_input = Input(shape=inputShape)
    print(X_input.shape)

    X = ZeroPadding2D((1,1))(X_input)
    X = Conv2D(32, kernel_size=2, strides=1, padding="same")(X)
    X = BatchNormalization(axis=3,name="zero_batch_norm")(X)
    X = Activation("relu")(X)


    X = MaxPooling2D(pool_size = (2,2))(X)

    print(X.shape)
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(64, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="first_batch_norm")(X)
    X = Activation("relu")(X)
    
    print(X.shape)
    X = Conv2D(128, kernel_size = 4, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3, name="2_batch_norm")(X)
    X = Activation("relu")(X)

    X = MaxPooling2D(pool_size=(2,2))(X)

    print(X.shape)
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(186, kernel_size=3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="3_batch_norm")(X)
    X_ = Activation("relu")(X)

    print(X.shape)
    X = Conv2D(256, kernel_size=4, strides=1, padding="valid")(X_)
    X = BatchNormalization(axis=3, name="4_batch_norm")(X)
    X_0 = Activation("relu")(X)

    print(X.shape)
    X = ZeroPadding2D((1,1))(X_)
    X = Conv2D(256, kernel_size = 3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis =3, name="5_batch_norm")(X)
    X_1 = Activation("relu")(X)

  

    X = Concatenate()([X, X_1])
    print("post concat")
    print(X.shape)

    
    #maybe try dropfilter early on
    X = Conv2D(1024, kernel_size=4, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="6_batch_norm")(X)
    X = Activation("relu")(X)
    print(X.shape)
    X = Flatten()(X)

    X  = Dropout(0.25)(X)

    X = Dense(128, activation="relu")(X)

    X = Dropout(0.275)(X)

    X_ = Dense(64, activation="relu")(X)

    X = Dropout(0.25)(X)
    
    X = Dense(64, activation="relu")(X)
    

    X = Dense(64)(X)

    X = Dropout(0.2)(X)

    X = Activation("relu")(X+X_)

    X = Dropout(0.2)(X)

    X = Dense(outputShape, activation="softmax")(X)

    model = Model(inputs=X_input, outputs = X)

    return model
"""
    




print("beginning loop")

#import imgaug as ia  ## was here so I could make sure was unrwapping images correctly




#model loading
model = classification_model((32,32,3), 10)

if (loadModel):
    try:
        model.load_weights(path)
        print("model successfully loaded")
    except:
        print("unable to load weights into model")
model.compile(optimizer="Adam",loss='categorical_crossentropy', metrics = ["accuracy", get_f1])
print(model.summary())

individualTestSize = 0 
for file_num in range(1,6):
    information = unpickle(file+str(file_num))

    X = information[b'data']
    y = information[b'labels']


    y = to_categorical(y)

##    image = X[0]
##    image = np.reshape(image, newshape = (32,32,3), order="F")
##    image =np.transpose(image, axes=(1,0,2))
##    
##    print(type(image))
##    print(image.shape)
##    ia.imshow(image)

    X = [np.transpose(np.reshape(image, newshape=(32,32,3), order="F"), axes=(1,0,2)) for image in X]

    #ia.imshow(X[0])  this was there to make sure was unwrapping images correctly

    #image order is same each time, so can just divide into valSet on the fly

    #converting to np array, scaling data, creating validation set
    
    valSize = 512
    X = np.array(X)
    X = X/255
   
    
    X_train, X_val = X[:-valSize, :, : , : ], X[-valSize:, :, :  ,:]
    y_train, y_val = y[:-valSize, :], y[-valSize:, :]

    if (file_num==1):
        X_trains = X_train
        y_trains = y_train
        X_vals = X_val
        y_vals = y_val
        individualTestSize = X_train.shape[0]
    else:
        X_trains = np.concatenate((X_trains, X_train))
        y_trains = np.concatenate((y_trains, y_train))
        X_vals = np.concatenate((X_vals, X_val))
        y_vals = np.concatenate((y_vals, y_val))
#we're going to combine, if didn't fit in memory could train with variable validation set or combine all val sets then repeat loop
print(X_trains.shape)
print(y_trains.shape)
print(X_vals.shape)
print(y_vals.shape)


#data augmentatoin to try n prevent overfitting
from random import randint
from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(
        rescale = 1/255,
        rotation_range = 15,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.65, 1.35))

#for plotting them curves
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.ion()
listOfValAcc = list()
listOfRegAcc = list()
listOfRegCost = list()
listOfValCost = list()
for j in range(5):
    segment = False
    if (segment):
        for i in range(1,6):
            print("decimal done is " + str(j/4+(i-1)/5/5))
            X_train = X_trains[(i-1)*individualTestSize:i*individualTestSize,:,:,:]

          
           
            y_train = y_trains[(i-1)*individualTestSize:i*individualTestSize,:]
            if (randint(1,4)):#changed to always do this one
                print("generator")
                train_gen = generator.flow(X_train, y_train, batch_size=64)
                
                
                
                history = model.fit(train_gen, verbose = 2, batch_size=64, callbacks = [model_checkpoint_callback], validation_data = (X_vals, y_vals), epochs=2)

                train_gen = generator.flow(X_train, y_train, batch_size=64)
                
                
                
                history = model.fit(train_gen, verbose = 2, batch_size=64, callbacks = [model_checkpoint_callback], validation_data = (X_vals, y_vals), epochs=2)
                 
            else:
                print("regular")

                history = model.fit(X_train, y_train, verbose = 2, batch_size=64, callbacks = [model_checkpoint_callback], validation_data = (X_vals, y_vals), epochs=4)
            
            listOfValAcc += history.history["val_accuracy"]
            listOfValCost += history.history["val_loss"]
            listOfRegCost += history.history["loss"]
            listOfRegAcc += history.history["accuracy"]

            finalX = min(len(listOfRegAcc), len(listOfValCost))

            stepSizeVal = finalX/len(listOfValCost)
            stepSizeReg = finalX/len(listOfRegCost)

            x_val = [i * stepSizeVal for i in range(len(listOfValCost))]
            x_reg = [i * stepSizeReg for i in range(len(listOfRegAcc))]

            plt.plot(x_val, listOfValAcc, color="green")
            plt.plot(x_val, listOfValCost, color="red")
            plt.plot(x_reg, listOfRegCost, color="yellow")
            plt.plot(x_reg, listOfRegAcc, color="blue")

            patches = [mpatches.Patch(color="green", label="val_acc"), mpatches.Patch(color="red",label="val_cost"), mpatches.Patch(color="yellow", label="reg_cost"), mpatches.Patch(color="blue", label="reg_acc")]
            plt.legend(handles=patches)
            plt.pause(0.00000001)
            
            plt.show()
            print(model.evaluate(X_trains, y_trains, verbose=0))
    else:
        train_gen = generator.flow(X_trains, y_trains, batch_size=64)
        history = model.fit(train_gen, verbose = 2, batch_size=64, callbacks = [model_checkpoint_callback], validation_data = (X_vals, y_vals), epochs=2)
        print(history)
            
        
        
    """

    generator = ImageDataGenerator(
        rotation_range = 30,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.65, 1.35))
    def augment_data(X):
        if (randint(0,1)):

    """
    
    
    
