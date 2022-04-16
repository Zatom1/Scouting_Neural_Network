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
import pandas as pd
import scipy as sp

from oauth2client.service_account import ServiceAccountCredentials
import gspread
import json
import tkinter as tk

scouting_sheet_name = '4786-69'

scopes = [
'https://www.googleapis.com/auth/spreadsheets',
'https://www.googleapis.com/auth/drive'
]
credentials = ServiceAccountCredentials.from_json_keyfile_name("C:/Users/zidda/.spyder-py3/Programs/Scouting_Neural_Network/Google API management/scoutingoutput-a55f2dc8af33.json", scopes) #access the json key you downloaded earlier 
file = gspread.authorize(credentials) # authenticate the JSON key with gspread
scoutingsheet = file.open("scouting test") #open sheet
scoutingsheet = scoutingsheet.sheet1 #replace sheet_name with the name that corresponds to yours, e.g, it can be sheet1


# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass


class ScanNotStraight(Error):
    """Raised when a data analyst scans a scouting sheet incorrectly"""
    pass

#print(googleSheet)

#from sklearn.datasets import fetch_openml 

#mnist = fetch_openml('mnist_784', return_X_y=True)

#file = 'mnist_784' 
#newfilename = 'gpsr_model.sav'
#pickle.dump(mnist, open(file, 'wb'))

  

mnist_load = 'mnist_784'
mnist = pickle.load(open(mnist_load, 'rb'))

img_size = 28

def img_crop(img, x, y, w, h):
    im_crop = img.crop((x, y, x+w, y+h))
    return im_crop

def img_to_arr(name, return_bool):
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
    
    #plt.imshow(disp)
    #print(np_img.reshape(1, img_size**2))

    predicted_img = np_img.reshape(1, img_size**2)
    if return_bool == True:
        return predicted_img
    else:
        return disp

def find_sheet(name, return_arr):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(THIS_FOLDER, name)

    imgopen = Image.open(img, mode="r").convert('L')
    img2 = os.path.join(THIS_FOLDER, 'file.jpg')

    imgopen.save(img2)
    imgopen2 = Image.open(img2, mode="r").convert('L')
    imgopen3 = imgopen2.resize((1691, 1120))
    #print(disp)
    #print(np_img.reshape(853, 703))
    #plt.imshow(np.asarray(imgopen2))
    #print(np.asarray(imgopen2)[10].sum()/1691)
    if not return_arr:
        return imgopen2
    else:
        return np.asarray(imgopen3)

def cut_top_black(img_name):
    img_arr = find_sheet(img_name, True)
    base = None
    sum_arr = None
    #print(img_arr[10000, 10])
    
    try:
        for i in range(50):
            if img_arr[i].sum()/1691 > 230 and img_arr[i].sum()/1691 < 200 :
                base = i
                sum_arr = img_arr[i].sum()/1691
                
                for i in range(base):
                    img_arr = np.delete(img_arr, 0, axis=0)
                    
                break
            elif i == 49:
                raise ScanNotStraight
            else:
                sum_arr = 0;
                break
                
    except ScanNotStraight:
        print("The scan is not straight enough to be read by the program")
    finally:
        if sum_arr == None:
            print("I have no idea how you even got here. Data analyst moment")
    #print(img_arr[0].sum()/1691)
    #print(img_arr) 
    #plt.imshow(img_arr)
    return img_arr

def cut_top_white(img_arr):
    #img_arr = find_sheet(img_name, True)
    base = None
    sum_arr = None
    try:
        for i in range(50):
            if img_arr[i].sum()/1691 < 205:
                base = i
                sum_arr = img_arr[i].sum()/1691
                
                for i in range(base):
                    img_arr = np.delete(img_arr, 0, axis=0)
                    
                break
            elif i == 49:
                raise ScanNotStraight
                
    except ScanNotStraight:
        print("The scan is not straight enough to be read by the program")
    finally:
        if sum_arr == None:
            print("I have no idea how you even got here. Data analyst moment")
    #print(img_arr[0].sum()/1691)
    #print(img_arr) 
    return img_arr

def cut_left_real(img_arr):
    #img_arr = find_sheet(img_name, True)
    base = None
    pil_image=Image.fromarray(img_arr)
    base = None
    for i in range(200, 300):
        if img_arr[400, i] < 100:
            base = i
            break
    """
    for i in range(1300):
        for w in range(base):
            img_arr[i] = np.delete( img_arr[i], 0)
  """ 
    cropped = img_crop(pil_image, base, 0, 1691-base, 1120)
    img_arr = np.asarray(cropped)
    #print(img_arr[0].sum()/1691)
    #print(base)
    return img_arr

def straighten_image(cut):
    split_left = None
    split_right = None
    for i in range(20):
        #print(cut[i].sum()/1691)
        if cut[i].sum()/1691 <= 230:
            ary = cut[i].reshape(89, 19)
            plt.imshow(ary)
            split_left = np.array_split(ary, 2)[0]
            split_right = np.array_split(ary, 2)[1]
            print(split_left.sum(), " - ", split_right.sum())
            #for w in range(220, len(cut[i]-1)-200):
                #print(cut[i, 1140])

    #for i in range(1):
        #img = Image.fromarray(cut)
        #imgR = img.rotate(50)
        #cut = np.asarray(imgR)
        #img.show()

        
    if split_left.sum() - split_right.sum() > 2500:
        img = Image.fromarray(cut)
        imgR = img.rotate(-1)
        cut = np.asarray(imgR)
        print("-1")
        
        if cut[0].sum()/1691 <= 230:
            ary = cut[i].reshape(89, 19)
            #plt.imshow(ary)
            split_left = np.array_split(ary, 2)[0]
            split_right = np.array_split(ary, 2)[1]
        else:
            None
        
        print(split_left.sum(), " - ", split_right.sum())

    elif split_left.sum() - split_right.sum() < -2500:
        img = Image.fromarray(cut)
        imgR = img.rotate(1)
        cut = np.asarray(imgR)            
        print("1")
            
        if cut[0].sum()/1691 <= 230:
            ary = cut[i].reshape(89, 19)
            #plt.imshow(ary)
            split_left = np.array_split(ary, 2)[0]
            split_right = np.array_split(ary, 2)[1]
        else:
            None
            
        print(split_left.sum(), " - ", split_right.sum())

    else:
        None
        
    ary = cut[0].reshape(89, 19)
    #plt.imshow(ary)
    split_left = np.array_split(ary, 2)[0]
    split_right = np.array_split(ary, 2)[1]
        
    if split_left.sum() - split_right.sum() > 2500:
        img = Image.fromarray(cut)
        imgR = img.rotate(-1)
        cut = np.asarray(imgR)
        print("-1")
        
        if cut[0].sum()/1691 <= 230:
            ary = cut[i].reshape(89, 19)
            #plt.imshow(ary)
            split_left = np.array_split(ary, 2)[0]
            split_right = np.array_split(ary, 2)[1]
        else:
            None
        
        print(split_left.sum(), " - ", split_right.sum())

    elif split_left.sum() - split_right.sum() < -2500:
        img = Image.fromarray(cut)
        imgR = img.rotate(1)
        cut = np.asarray(imgR)            
        print("1")
            
        if cut[0].sum()/1691 <= 230:
            ary = cut[i].reshape(89, 19)
            #plt.imshow(ary)
            split_left = np.array_split(ary, 2)[0]
            split_right = np.array_split(ary, 2)[1]
        else:
            None
            
        print(split_left.sum(), " - ", split_right.sum())

    else:
        
        None
    img = Image.fromarray(cut)
    img.show()
    
    cut = np.asarray(img)
    
    return cut

def align_img(img_name):
    cut_top_img = cut_top_black(img_name + '.jpg')
    

    cut_top_white_img = cut_top_white(cut_top_img)

    cut_left = cut_left_real(cut_top_white_img)

    #plt.imshow(cut_left)
    return cut_left

X, y = mnist

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) # splits data into training and testing sets


#mlp = MLPClassifier(random_state=2, hidden_layer_sizes=(100,100, 100), solver='adam', max_iter=1500)
#mlp.fit(X, y) # code to generate new neural network

neural_network = 'scouting_nn_3L_FULL'
mlp = pickle.load(open(neural_network, 'rb')) # opens neural network file to improve load speeds


#file = 'scouting_nn_3L_FULL' 
#pickle.dump(mlp, open(file, 'wb')) # code to save file of neural network

def predict(file_name):
    predict = img_to_arr(file_name + '.jpg', True)
    return mlp.predict(predict)
    
def predict_img(file_name):
    #predict = img_to_arr(file_name + '.jpg')
    return mlp.predict(file_name)


sheet = find_sheet(scouting_sheet_name + '.jpg', False)

sheet_aligned = align_img(scouting_sheet_name)

sheet_img = Image.fromarray(sheet_aligned)

sheet_img.show()

def enhance(img_arr):
    
    for i in range(img_size):
        for w in range(img_size):
            if img_arr[i, w] > 66:
                img_arr[i, w] = 0
            else: 
                img_arr[i, w] = 255
    #print(img_arr)
    return img_arr

            
def find_num(img_name, x, y, w, h):
    num_crop = img_crop(sheet, x, y, w, h)
    num_crop.save(img_name + '.jpg')
    
    num_arr = img_to_arr(img_name + '.jpg', False)
    enhance(num_arr)
    
    reshape_num_arr = num_arr.reshape((28, 28))
    for i in range(img_size):
        if reshape_num_arr[i].sum() == 0 and i > img_size/2:
            num_crop = img_crop(sheet, x, y-i, w, h)
        elif reshape_num_arr[i].sum() == 0 and i <= img_size/2:
            num_crop = img_crop(sheet, x, y+i, w, h)
    
    num_crop.save(img_name + '.jpg')
    num_arr = img_to_arr(img_name + '.jpg', False)
    enhance(num_arr)

    image = Image.fromarray(num_arr)
    image.save(img_name + '.jpg')
    plt.imshow(num_arr)
    prediction = predict(img_name)
    print(prediction)
    return prediction


def read_sheet():
    num_high_left = find_num('num_high', 938, 1034, 80, 70)
    num_high_right = find_num('num_high', 938+80, 1034, 80, 70)
    total_high = (int(num_high_left[0])*10) + int(num_high_right[0])
    
    
    num_low_left = find_num('num_low', 1115, 1034, 80, 70)
    num_low_right = find_num('num_low', 1115+80, 1034, 80, 70)
    total_low = (int(num_low_left[0])*10) + int(num_low_right[0])
    
    print(total_high)
    #print(total_low)
    
    num_missed_left = find_num('num_missed', 1285, 1034, 80, 70)
    num_missed_right = find_num('num_missed', 1285+80, 1034, 80, 70)
    
    #rung_reached = find_num('rung_num', 488, 547, 120, 35)


    scoutingsheet.update_cell(1, 2, total_high) #updates row 2 on column 3

window=tk.Tk()
window.title(" FEAR Scouting Interface ")
window.geometry("1000x600")
newlabel = tk.Label(text = " Visit Pythonista Planet to improve your Python skills ")
newlabel.grid(column=0,row=0)
button_name = tk.Button(window, text = "some text")
button_name.grid(column=1,row=0)
button_name.bind("<Button-1>", read_sheet()) 

window.mainloop()

