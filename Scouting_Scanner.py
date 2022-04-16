# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:40:11 2022

@author: zidda
"""
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

from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
from tkinter.ttk import Progressbar


import time
from PIL import Image



path ="C:/Users/zidda/.spyder-py3/Programs/Scouting_Neural_Network/Program Directory"
#we shall store all the file names in this list
filelist = []

for root, dirs, files in os.walk(path):
	for file in files:
        #append the file name to the list
		filelist.append([os.path.join(root,file)])

namelist = []

#print all the file names
for name in filelist:
    strname = str(name)
    dash = '-'
    if strname[-9] == '-':

        print(strname[-9])
        arr_short = strname[-13:-2] # , -11:-4
        print(arr_short)
        #print(name[-11])
        namelist.append(strname[-13:-2])

    
ws=tk.Tk()
ws.title(" FEAR Scouting Interface ")
ws.geometry("315x80")
instructions = tk.Label(text = " Select file from directory w/o '~'  ->  ")
instructions.grid(column=0,row=0)

#file_selector = tk.OptionMenu(window,tk.IntVar(),for name in namelist)
#file_selector.grid(column=1,row=0)

#Define a function to close the window
def close_win():
   ws.destroy()

progress_bar = Progressbar(
    ws, 
    orient=HORIZONTAL, 
    length=300, 
    mode='indeterminate'
    )
        
def open_file():
    file_path = askopenfile(mode='r', filetypes=[('pls enter jpg only', '*jpg')])
    print(file_path)

    string_path = str(file_path)
    print(string_path[-40:-33])
    progress_bar.start()
    global scouting_sheet_name
    scouting_sheet_name = str(string_path[-40:-33])
    progress_bar.destroy()
    Label(ws, text='File Uploaded Successfully!', foreground='green').grid(row=4, columnspan=3, pady=10)
    print(scouting_sheet_name)
    if file_path is not None:
        pass
   
    
sheet_upload = Label(
    ws, 
    text='Upload scouting sheet scan'
    )
sheet_upload.grid(row=0, column=0, padx=10)

choose = Button(
    ws, 
    text ='Choose File', 
    command = lambda:open_file()
    ) 
choose.grid(row=0, column=1)



close = Button(
    ws, 
    text='Read Sheet', 
    command=close_win
    )
close.grid(row=3, columnspan=3, pady=10)

ws.mainloop()



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

#scouting_sheet_name = '4786-69'


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

    if not return_arr:
        return imgopen2
    else:
        return np.asarray(imgopen3)

def cut_top_black(img_name):
    img_arr = find_sheet(img_name, True)
    base = None
    sum_arr = None
    
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
    cut = np.asarray(img)
    
    return cut

def align_img(img_name):
    cut_top_img = cut_top_black(img_name + '.jpg')

    cut_top_white_img = cut_top_white(cut_top_img)

    cut_left = cut_left_real(cut_top_white_img)

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
    
sheet = find_sheet(scouting_sheet_name + '.jpg', False)

sheet_aligned = align_img(scouting_sheet_name)

sheet_img = Image.fromarray(sheet_aligned)

#sheet_img.show()

def enhance(img_arr):
    
    for i in range(img_size):
        for w in range(img_size):
            if img_arr[i, w] > 66:
                img_arr[i, w] = 0
            else: 
                img_arr[i, w] = 255
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
    
    num_missed_left = find_num('num_missed', 1285, 1034, 80, 70)
    num_missed_right = find_num('num_missed', 1285+80, 1034, 80, 70)
    total_missed = (int(num_missed_left[0])*10) + int(num_missed_right[0])

    
    #rung_reached = find_num('rung_num', 488, 547, 120, 35)

    print(total_high)
    print(total_low)
    print(total_missed)
    
    scoutingsheet.update_cell(2, 1, total_high) 
    scoutingsheet.update_cell(2, 2, total_low) 
    scoutingsheet.update_cell(2, 3, total_missed) 

read_sheet()
