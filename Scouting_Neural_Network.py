# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:40:11 2022

@author: zidda
"""
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import os
from skimage import color
from skimage import io


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
img = os.path.join(THIS_FOLDER, 'oneNNCompresses.jpg')

imgopen = Image.open(img, mode="r").convert('L')
#imgopen.show()

small_img1 = imgopen.resize((8, 8))

small_img2 = small_img1.resize((8, 8))

np_img = np.asarray(small_img2) 


#read_image = io.imread(img)

#greyIMG = color.rgb2gray(read_image)
#io.imsave("greyIMG.png",img)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

plt.imshow(imgopen)

#plt.show()

for i in range(len(np_img)):
    for w in range(len(np_img)): 
        #print(np_img[i, w])
        np_img[i, w] = 255 - np_img[i, w]
        
np_arr_1d = np.empty(shape=[1, 64])

for i in range(len(np_img)):
    for w in range(len(np_img)):
       
        np.append(np_arr_1d, (float(np_img[i, w])))

print(small_img2)
#small_img.show()



x = X_train[1]
"""
for j in range(len(incorrect)):

    plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    print("true value:", incorrect_true[j])
    print("predicted value:", incorrect_pred[j])
"""

#print(mlp.score(X_test, y_test))
print(np_img.reshape(1, 64))
plt.xticks(())
plt.yticks(())
     
print(mlp.predict(np_img.reshape(1, 64)))

