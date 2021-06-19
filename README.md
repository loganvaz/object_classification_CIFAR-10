# object_classification_CIFAR-10
Using dataset from paper (Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.) provided by website (https://www.cs.toronto.edu/~kriz/cifar.html) for practice with CNNs


Alright, model data (weights and such) declared as too large

Elements of objectClassification.py

createFolders: create a folder, or if already there return path

unpickle: for data extraction

y_convert: to convert text->one_hot encoding and back

checkpoint: to save model based on best accuracy

get_f1: additional metric (taken from old keras source code)

model: we have convolutions followed by normalization then activations, also maxPooling and padding
       we have multiple convolution layers contributing to final flatten directly and have a series of dense layers in the final stage (using X + X_ to try n ensure that model length is helpful rather than harming)
