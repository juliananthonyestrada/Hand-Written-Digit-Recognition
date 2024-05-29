import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# loads data set
mnist = tf.keras.datasets.mnist

# split data into training and testing set - we use the training set to train the model and we then evaluate the model on the testing set
# x data is the images and y data is the classifications
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# scale data so that values are between 0 and 1 (we are normalizing the pixels)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# reshape the data to fit the model (adding a channel dimension)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# initialize the neural network model
cnn_model = tf.keras.models.Sequential()

# add convolutional layers
cnn_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# flatten the convolutional output to feed into dense layers
cnn_model.add(tf.keras.layers.Flatten())

# add dense layers
cnn_model.add(tf.keras.layers.Dense(256, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(256, activation='relu'))

# add output layer -> 10 layers to represent 10 possible output digits
# softmax provides a 'probability' to each neuron -> greatest value is output
cnn_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit data to the model
cnn_model.fit(x_train, y_train, epochs=3)

# save the model to the directory
cnn_model.save('handwritten_model_cnn.keras')

cnn_model = tf.keras.models.load_model('handwritten_model_cnn.keras')

'''
loss, accuracy = cnn_model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
'''

# declare iterator
image_number = 1

# while the image exists
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # read and invert the image
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        img = img.reshape(-1, 28, 28, 1)  # reshape to match model's input shape
        prediction = cnn_model.predict(img)
        # predict
        print(f"This digit is probably a {np.argmax(prediction)}")
        # show image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        # handle exception
        print(f"There is a problem with the image: {e}")
    finally:
        # go to next image
        image_number += 1
