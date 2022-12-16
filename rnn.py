# multi-class classification with Keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import pandas
from keras.models import Sequential
from keras.layers import Activation, Dense
plt.style.use('ggplot')

# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# # load dataset
# dataframe = pandas.read_csv("dataset-manda-terus.csv",)
# dataset = dataframe.values
# ilabel = 1000
# jclass = 6
# X = dataset[:,0:ilabel].astype(float)
# Y = dataset[:,ilabel]

dataframe = pandas.read_csv("dataset-manda-terus.csv")
dataset = dataframe.values
ilabel = 1000
jclass = 6
X = dataset[:,0:ilabel].astype(float)
Y = dataset[:,ilabel]


# classes = ['Sepatu Basket', 'Sepatu Slip On', 'Sandals', 'Pantofel', 'High Heels', 'Sepatu Running']
# # Binarize the output
# y_bin = label_binarize(y, classes=classes)
# n_classes = y_bin.shape[1]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
model.add(Dense(128, input_dim=1000, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# model.add(Dense(8, input_dim=ilabel, activation='relu'))
model.add(Dense(jclass, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nepochs = 200
nbatch = 5

# ------------ menggunakan packages
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.20)

model.fit(X_train, y_train, epochs=nepochs, batch_size=nbatch)
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# create model
model = Sequential()

model.add(Dense(128, input_dim=1000, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# model.add(Dense(8, input_dim=ilabel, activation='relu'))
model.add(Dense(jclass, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nepochs = 200
nbatch = 5

# ------------ menggunakan packages
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.20)

model.fit(X_train, y_train, epochs=nepochs, batch_size=nbatch)
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# save model
model.save("my_model")
model.save_weights("model.h5")
y_score = model.predict(X_test)
y_score
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100) + '%')

# Plotting and estimation of FPR, TPR
fpr = dict()
tpr = dict()
roc_auc = dict()


y_score = model.predict(X_test)