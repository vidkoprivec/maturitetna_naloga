import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


model = load_model('digits.model', compile = True)


slika = cv.imread('test.png')[:,:,0]
slika = np.invert(np.array([slika]))
prediction = model.predict(slika)
print(np.argmax(prediction))
plt.imshow(slika[0], cmap=plt.cm.binary)
plt.show()

