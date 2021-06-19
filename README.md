# object_classification_CIFAR-10
Using dataset from paper (Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.) provided by website (https://www.cs.toronto.edu/~kriz/cifar.html) for practice with CNNs


Alright, model data (weights and such) declared as too large

# Elements of objectClassification.py

createFolders: create a folder, or if already there return path

unpickle: for data extraction

y_convert: to convert text->one_hot encoding and back

checkpoint: to save model based on best accuracy

get_f1: additional metric (taken from old keras source code)

model: we have convolutions followed by normalization then activations, also maxPooling and padding
       we have multiple convolution layers contributing to final flatten directly and have a series of dense layers in the final stage (using X + X_ to try n ensure that model length is helpful rather than harming)

# Elements of saveBestModel.py

we define stuff custom (valSize, inputShape, numClasses) as it was easier

we have two model paths and two models (firstModel and secondModel)

we have the unpickle function to parse the data

we load the weights in both folders (one where model from previous trianing was stored, one where best trianing model was stored) into model and model2, trusting that the weights will only load into appropriate models (could be wrong, but believe that assumption is just Weights(1) != Weights(2) which seems reasonable so long as we're changing stuff other than just Dropout)

next we look at how each perform on teh validation set, figiure out which is better, move it to the finalModel folder and print whether it was firstModel or secondModel code that needs to be kept
