import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Input, Reshape, Dense,Conv2D,MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from tqdm import tqdm


print(tf.test.is_built_with_cuda())
print(tf.test.is_built_with_gpu_support())
print(tf.test.is_gpu_available())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def preprocess_results(results):
    y = []
    for i in results:
        array = np.zeros(10,np.uint8)
        array[i]=1
        y.append(array)
    return np.array(y)

#if want to pass the test labels in evaluating, uncomment the following line
#y_test=preprocess_results(y_test)
y_train=preprocess_results(y_train)

#Creating model or loading
if(os.path.exists("model.h5")):
    model = keras.models.load_model("model.h5")
else:
    model = keras.Sequential()
    model.add(Input((28,28)))
    model.add(Reshape((28,28,1)))
    model.add(Conv2D(32,1,1,activation='relu'))
    model.add(Conv2D(32,1,1, activation='relu'))
    model.add(MaxPooling2D(1,1))
    model.add(Conv2D(64,3,3,activation='relu'))
    model.add(Conv2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D(3,3))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=8,batch_size=16)
    print()

    
def get_prediction(image):
    image = np.resize(image,(1,28,28,1))
    prediction = model.predict(image)
    prediction = prediction[0]
    prediction = np.where(prediction == np.amax(prediction))[0]
    return prediction

def get_prediction_list(list):
    results = []
    for i in tqdm(list):
        results.append(np.array(int(get_prediction(i))))
    return np.array(results)

# Creating confussion matrix
predictions = get_prediction_list(x_test)
print(type(predictions))
print(type(y_test))
cm = confusion_matrix(y_test,predictions)
cmd = ConfusionMatrixDisplay(cm)
cmd.plot()
plt.show()


