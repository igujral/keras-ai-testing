import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling1D
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# Data Setup

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

# Visualize Data

num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(32,32))
for i in range (0, num_classes):
    sample = x_train[i]
    ax[i].imshow(sample)
    ax[i].set_title('Label {}'.format(i))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Normalize & Reshape Data
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape)
print(y_train.shape)

# Make NN

model = Sequential()
model.add(Dense(units=512, input_shape=(3072,), activation='relu'))
keras.layers.MaxPooling1D (
   pool_size = 3, 
   strides = None, 
   padding = 'valid', 
   data_format = 'channels_last'
)
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x=x_train, y=y_train, batch_size=512, epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
