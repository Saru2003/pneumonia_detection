import os
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# list files in the dataset directory
input_dir = "C:\\Users\\Asus\\Documents\\code\\cloud_project\\archive\\chest_xray"
for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# labels and image size
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

# load and preprocesses data
def get_training_data(data_dir):
    data = []
    labels_data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) 
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # resizing images to 150x150
                data.append(resized_arr) 
                labels_data.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return np.array(data), np.array(labels_data)

# load train, test, and validation data
train_data, train_labels = get_training_data(os.path.join(input_dir, 'train'))
test_data, test_labels = get_training_data(os.path.join(input_dir, 'test'))
val_data, val_labels = get_training_data(os.path.join(input_dir, 'val'))

# plot distribution of labels
sns.set_style('darkgrid')
sns.countplot([labels[i] for i in train_labels])

# vis. sample images
plt.figure(figsize=(5, 5))
plt.imshow(train_data[0], cmap='gray')
plt.title(labels[train_labels[0]])

plt.figure(figsize=(5, 5))
plt.imshow(train_data[-1], cmap='gray')
plt.title(labels[train_labels[-1]])

# normalize, reshape data
x_train = train_data / 255.0
x_val = val_data / 255.0
x_test = test_data / 255.0

x_train = x_train.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)

y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

# model defn
model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

# train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=12,
                    validation_data=(x_val, y_val),
                    callbacks=[learning_rate_reduction])

# evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss of the model is - {loss}")
print(f"Accuracy of the model is - {accuracy * 100}%")
model.save('pneumonia_detection_model.h5')
