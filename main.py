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

'''
# initialize the neural network model
nn_model = tf.keras.models.Sequential()

# add a later to the model -> we are flattening the 28 x 28 pixel grid into one continuous layer of pixels
nn_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# add layers to the model -> Dense layer is the most basic layer (google neural network graph) -> with 128 neurons each
nn_model.add(tf.keras.layers.Dense(128, activation='relu'))
nn_model.add(tf.keras.layers.Dense(128, activation='relu'))

# add output layer -> 10 layers to represent 10 possible output digits
# softmax provides a 'probability' to each neuron -> greatest value is output
nn_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile the model
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit data to the model
nn_model.fit(x_train, y_train, epochs=3)

# save the model to the directory
nn_model.save('handwritten_model.keras')
'''

# lines 19-42 train the model and lines 46-52 evaluate
# to evaluate the model comment out 19-42 and to train comment out 46-52

nn_model = tf.keras.models.load_model('handwritten_model.keras')

'''
loss, accuracy = nn_model.evaluate(x_test, y_test)

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
        prediction = nn_model.predict(img)
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
