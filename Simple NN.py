import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('traincomplex.csv')
np.random.shuffle(train_df.values)

print(train_df.head())

model = keras.Sequential([
	keras.layers.Dense(1024, input_shape=(2,), activation='relu'),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, validation_split = 0.1, batch_size=32, epochs=20, shuffle=True, verbose=1)

test_df = pd.read_csv('testcomplex.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("PREDICTION SET")
predictions = model.predict(x=test_x, batch_size=10, verbose=1)
wrong_array_x, wrong_array_y = [], []
for i in range(len(predictions)):
    if predictions[i][0] > 0.5:
        new_prediction = 0.0
    elif predictions[i][1] > 0.5 :
        new_prediction = 1.0
    else: 
        new_prediction = 3.0
    if new_prediction != test_df.color.values[i] and new_prediction != 3.0: 
        train_df.loc[len(train_df.index)] = [test_df.x.values[i], test_df.y.values[i], 2]
    elif new_prediction == 3.0:
        train_df.loc[len(train_df.index)] = [test_df.x.values[i], test_df.y.values[i], 3]
       #train_df = train_df.concat(df2, ignore_index=True)
    else: 
        train_df.loc[len(train_df.index)] = [test_df.x.values[i], test_df.y.values[i], 4]


color_dict =  {0:'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'black'}
train_df['color'] = train_df.color.apply(lambda x: color_dict[int(x)])

plt.scatter(train_df.x, train_df.y, color=train_df.color, s=1)
plt.show()

model.evaluate(test_x, test_df.color.values)