import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.preprocessing import image
from PIL import Image

np.random.seed(3)

"""# Data"""

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np.squeeze(y_train)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

classify_dict = {0: 'airplane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

"""# Visualize Examples"""

num_classes = 10
f, ax = plt.subplots(1, num_classes+1, figsize=(32,32))

for i in range(0, num_classes):
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title("Label: {}".format(classify_dict[i]), fontsize=16)
  

for i in range(10):
  print(y_train[i])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

for i in range(10):
  print(y_train[i])

"""# Prepare Data"""

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape Data
x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3)
print(x_train.shape)

"""# Create Model - Fully Connected Neural Network"""

model = Sequential()

model.add(Dense(units=128, input_shape=(3072,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

"""# Train"""

batch_size = 512
epochs=10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

"""# Evaluate"""

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

# Single Example
random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(classify_dict[y_sample_pred_class], classify_dict[y_sample_true]), fontsize=16)
plt.imshow(x_sample.reshape(32, 32, 3), cmap='gray')

# Helper function to classify an uploaded image
def classify_uploaded_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize the image to match the CIFAR-10 input size
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 32 * 32 * 3)  # Reshape the image for the model
    pred = model.predict(img)
    pred_class = np.argmax(pred)
    return classify_dict[pred_class]

# Example usage:
image_path = "cifar10_uploaded_test_imgs/deer1.jpeg"
predicted_class = classify_uploaded_image(image_path)
print("Predicted class:", predicted_class)

"""# Confusion Matrix"""
# Create a list of label names in the desired order
label_names = [classify_dict[i] for i in range(len(classify_dict))]

y_true_mapped = [label_names[idx] for idx in y_true]
y_pred_mapped = [label_names[idx] for idx in y_pred_classes]

confusion_mtx = confusion_matrix(y_true_mapped, y_pred_mapped, labels=label_names)

# Plot
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix');