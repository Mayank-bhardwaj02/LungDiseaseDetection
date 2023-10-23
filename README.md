# Lung Disease Classification using Convolutional Neural Networks (CNN)

## Overview

This project focuses on classifying lung diseases, specifically detecting pneumonia, using Convolutional Neural Networks (CNN). The architecture employs multiple convolutional layers, pooling layers, and dense layers to process and classify lung images.

## Data Preprocessing

The data has been preprocessed to fit the required input shape, ensuring optimal performance when fed into the model.

## Coming Soon
Training, evaluation, and detailed results of the model will be updated shortly.

## Acknowledgments
Special thanks to data source for providing the lung images dataset.



## Model Architecture

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

image_shape = (image_width, image_height, color_channels)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


