#inputs

valSize = 512#should be the same as objectClassification.py
inputShape, numClasses = (32,32,3), 10


#libraries
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import pickle
def createFolders(h, n):
    try:
        f = os.mkdir(h+"\\"+n)
        f.close()
    except:
        return h+"\\"+n
    return h+"\\"+n


path = createFolders(os.getcwd(), "finalModel")

otherPath = os.getcwd() + "\\modelInProgess"

os.chdir(otherPath)
print(os.listdir())
print(len(os.listdir()))


place1 = len(os.listdir())

goodModel1 = False

###models
from keras.layers import Concatenate, Activation, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, BatchNormalization, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.keras import regularizers

def firstModel(inputShape, outputShape):#this is the better model
    #def classification_model(inputShape, outputShape):
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

    courseX = Dropout(0.25)(courseX)

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


def secondModel(inputShape, outputShape):#this is the worse model (difference is level of dropout)
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

    courseX_ = Dropout(0.3)(courseX_)

    courseX = Dense(256, activation="relu")(courseX_)

    courseX = Dropout(0.4)(courseX)

    courseX = Dense(128)(courseX_)

    courseX = Dropout(0.3)(courseX)

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


    X = Dropout(0.375)(X)

    X = Dense(128, activation="relu")(X)

    X= Dropout(0.35)(X)
    
    X_ = X

    print(X_.shape)
   
    X = Dense(256, activation="relu")(X)

    X = Dropout(.3)(X)
    

    X = Dense(128)(X)

    X = Dropout(0.3)(X)


    print(X.shape)
    print(X_.shape)
    X = Activation("relu")(X+X_)

    #no dropout b/c already applied to each of these
    

    X = Dense(outputShape, activation="softmax")(X)

    model = Model(inputs=X_input, outputs = X)

    return model




#unpickle stuff
def unpickle(file):
    #import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d



if (place1!=0):
    try:
        model = firstModel(inputShape,numClasses)
        model.load_weights(otherPath+"\\model")
        goodModel1 = True
        firstModelIs = 1
    except:
        try:
            model = secondModel(inputShape,numClasses)
            model.load_weightS(otherPath+"\\model")
            goodModel1 = True
            firstModelIs = 2
        except:
            print("no model match")

else:
    print("nothing in modelInProgress")

os.chdir("..")
os.chdir(path)
print(os.listdir())
place2 = len(os.listdir())
print(place2)

import shutil
if (place2==0):
    print("nothing in final model")
    if (goodModel1):
        print("moving weights to final folder")
        os.chdir(otherPath)
        for file in os.listdir():
            print(file)
            shutil.copy(file,path)


else:
    try:
        model2 = firstModel(inputShape,numClasses)
        model2.load_weights(path + "\\model")
        secondModelIs = 1
    except:
        try:
            model2 = secondModel(inputShape,numClasses)
            model2.load_weights(path+"\\model")
            secondModelIs = 2
        except:
            print("no valid second model")
            if (goodModel1):
                print("moving weights to final folder")
                for file in os.listdir():
                    shutil.copy(file,path)
    if (goodModel1):
        os.chdir("..")
        file = os.getcwd()+"\\cifar-10-batches-py\\data_batch_"
        os.chdir("finalModel")
        for file_num in range(1,6):
            information = unpickle(file+str(file_num))

            X = information[b'data']
            y = information[b'labels']


            y = to_categorical(y)
            
            X = [np.transpose(np.reshape(image, newshape=(32,32,3), order="F"), axes=(1,0,2)) for image in X]
            X = np.array(X)
            X = X/255
           
            
            X_train, X_val = X[:-valSize, :, : , : ], X[-valSize:, :, :  ,:]
            y_train, y_val = y[:-valSize, :], y[-valSize:, :]

            if (file_num==1):
                X_vals = X_val
                y_vals = y_val
                individualTestSize = X_train.shape[0]
            else:
                X_vals = np.concatenate((X_vals, X_val))
                y_vals = np.concatenate((y_vals, y_val))
        model.compile(optimizer="Adam",loss='categorical_crossentropy', metrics = ["accuracy"])

        validationScore1 = model.evaluate(X_vals, y_vals, verbose=0)
        print(validationScore1)
        print("first score is from model" + str(firstModelIs))
        

        model2.compile(optimizer="Adam",loss='categorical_crossentropy', metrics = ["accuracy"])
        validationScore2 = model2.evaluate(X_vals, y_vals, verbose=0)

        print(validationScore2)

        print("second score is from model " + str(secondModelIs))

        print("if model1 has better accuracy, am moving it to finalModel folder")

        if (validationScore1[1]>validationScore2[1]):
            print("first is better, moving")
            os.chdir(otherPath)
            for file in os.listdir():
                print(file)
                shutil.copy(file,path)
            print("model need to keep is " + str(firstModelIs))

        else:
            print("each model same AND/OR current final is better")
            print("model need to keep is " + str(secondModelIs))
                    
  
