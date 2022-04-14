# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:40:11 2022

@author: zidda
"""
from PIL import Image
#from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from gsheets import Sheets
import pandas as pd

sheets = Sheets.from_files('client_secrets.json', '~/storage.json')
url = 'https://docs.google.com/spreadsheets/d/1qfHxtJ34MkHHXY2pvX3mCSq93THfva8_wRYrRD3eD3c/edit#gid=0'
s = sheets.get(url)

googleSheet = pd.DataFrame(s.sheets[0].values())

googleSheet[3, 5] = 25

print(googleSheet)

#from sklearn.datasets import fetch_openml 

#mnist = fetch_openml('mnist_784', return_X_y=True)

#file = 'mnist_784' 
#newfilename = 'gpsr_model.sav'
#pickle.dump(mnist, open(file, 'wb'))

mnist_load = 'mnist_784'
mnist = pickle.load(open(mnist_load, 'rb'))

img_size = 28

def img_to_arr(name):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(THIS_FOLDER, name)

    imgopen = Image.open(img, mode="r").convert('L')
    img2 = os.path.join(THIS_FOLDER, 'file.jpg')

    imgopen.save(img2)
    imgopen2 = Image.open(img2, mode="r").convert('L')

    small_img1 = imgopen2.resize((img_size, img_size))

    np_img = np.asarray(small_img1)
    
    for i in range(len(np_img)):
        for w in range(len(np_img)): 
            #print(np_img[i, w])
            np_img[i, w] = 255 - np_img[i, w]
            
    np_arr_1d = np.empty(shape=[1, img_size**2])

    for i in range(len(np_img)):
        for w in range(len(np_img)):
           
            np.append(np_arr_1d, (float(np_img[i, w])))
    
    
    disp = np_img.reshape(img_size, img_size)
    
    plt.imshow(disp)
    #print(np_img.reshape(1, img_size**2))

    predicted_img = np_img.reshape(1, img_size**2)
    return predicted_img


def find_sheet(name):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(THIS_FOLDER, name)

    imgopen = Image.open(img, mode="r").convert('L')
    img2 = os.path.join(THIS_FOLDER, 'file.jpg')

    imgopen.save(img2)
    imgopen2 = Image.open(img2, mode="r").convert('L')
    reshaped_img = imgopen2.resize((850, 700))

    np_img = np.asarray(reshaped_img)

    disp = np_img
    #print(disp)
    #print(np_img.reshape(853, 703))
    return reshaped_img


def img_crop(img, x, y, w, h):
    im_crop = img.crop((x, y, x+w, y+h))
    return im_crop

X, y = mnist

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) # splits data into training and testing sets


#mlp = MLPClassifier(random_state=2, hidden_layer_sizes=(100,100, 100), solver='adam', max_iter=1500)
#mlp.fit(X, y) # code to generate new neural network

neural_network = 'scouting_nn_3L_FULL'
mlp = pickle.load(open(neural_network, 'rb')) # opens neural network file to improve load speeds


#file = 'scouting_nn_3L_FULL' 
#pickle.dump(mlp, open(file, 'wb')) # code to save file of neural network

def predict(file_name):
    predict = img_to_arr(file_name + '.jpg')
    return mlp.predict(predict)
    
def predict_img(file_name):
    #predict = img_to_arr(file_name + '.jpg')
    return mlp.predict(file_name)

"""
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

for j in range(len(incorrect)):

    plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    print("true value:", incorrect_true[j])
    print("predicted value:", incorrect_pred[j])
"""

# above code checks and displays incorrect guesses

print(mlp.score(X_test, y_test))
#print(disp)

sheet = find_sheet('RevisedSheet.jpg')

sheet.show()

def  find_num(img_name, x, y, w, h):
    num_crop = img_crop(sheet, x, y, w, h)
    num_crop.save(img_name + '.jpg')
    num_arr = img_to_arr(img_name + '.jpg')
    prediction = predict(img_name)
    print(prediction)
    return prediction
    
"""
num_missed_crop = img_crop(sheet, 730, 640, 110, 60)
num_missed_crop.save('num_missed.jpg')
num_missed_arr = img_to_arr('num_missed.jpg')

num_low_crop = img_crop(sheet, 610, 640, 112, 55)
num_low_crop.save('num_low.jpg')
num_low_arr = img_to_arr('num_low.jpg')

num_missed = predict('num_missed')
num_low = predict('num_low')



#print(predict('six'))

print(num_missed, num_low)
"""

find_num('num_missed', 730, 640, 110, 60)

#pickle.dump(num_missed_crop, open('num_missed_crop.jpg', 'w')) # code to save file of neural network

#with open(num_missed_crop) as f:
 #   print(f)

#plt.imshow(num_missed_crop)
# 1, 2, 3, 5, 6, 7, 8, 9
#plt.imshow()





