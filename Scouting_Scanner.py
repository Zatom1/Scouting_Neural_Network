# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:40:11 2022

@author: zidda
"""
#from sklearn.datasets import load_digits
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd

from oauth2client.service_account import ServiceAccountCredentials
import gspread
import json
import tkinter as tk
import math
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
from tkinter.ttk import Progressbar

import io
from PIL import Image, ImageDraw, ImageTk

ws= Tk()
ws.title(" FEAR Scouting Interface ")
ws.geometry("650x420")
instructions = tk.Label(text = " Select file from directory w/o '~'  ->  ")
instructions.grid(column=0,row=0)


def submit():
    global team_num
    team_num=num_var.get()
     
    num_var.set("")
    
def close_win():
   ws.destroy()


num_var = tk.StringVar()

number_entry = tk.Entry(ws,textvariable = num_var, font=('calibre',10,'normal'))
number_entry.grid(row=5,column=1)

Button(
    ws,
    text='Set Team #',
    command=submit
).grid(column=1,row=6)

def run_all():
    scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name("C:/Users/zidda/.spyder-py3/Programs/Scouting_Neural_Network/Google API management/scoutingoutput-a55f2dc8af33.json", scopes) #access the json key you downloaded earlier 
    file = gspread.authorize(credentials) # authenticate the JSON key with gspread
    scouting_sheet_open = file.open("Final Worlds 2022 copy") #open sheet
    scoutingsheet = scouting_sheet_open.get_worksheet(1)
    #scoutingsheet = scoutingsheet.sheet_name #replace sheet_name with the name that corresponds to yours, e.g, it can be sheet1
    
      
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
    
    
    #mnist_load = 'mnist_784'
    #mnist = pickle.load(open(mnist_load, 'rb'))
    
    img_size = 28
    
    def img_crop(img, x, y, w, h):
        
        try:
            im_crop = img.crop((x, y, x+w, y+h))
            return im_crop

        except TypeError:
            print("img_crop was passed a non-number argument. args were: ", x, " - ", y, " - ", w, " - ", h)
            im_crop = Image
        finally:
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
    
    #X, y = mnist
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) # splits data into training and testing sets
    
    #mlp = MLPClassifier(random_state=2, hidden_layer_sizes=(100,100, 100), solver='adam', max_iter=1500)
    #mlp.fit(X, y) # code to generate new neural network
    
    neural_network = 'scouting_nn_3L_FULL'
    mlp1 = pickle.load(open(neural_network, 'rb')) # opens neural network file to improve load speeds
    
    #file = 'scouting_nn_3L_FULL' 
    #pickle.dump(mlp, open(file, 'wb')) # code to save file of neural network
    
    def predict(file_name):
        predict = img_to_arr(file_name + '.jpg', True)
        
        prediction1 = mlp1.predict(predict)
    
        return prediction1
    
    
        
    sheet = find_sheet(scouting_sheet_name + '.jpg', False)
    
    sheet_aligned = align_img(scouting_sheet_name)
    
    sheet_img = Image.fromarray(sheet_aligned)
    
    #sheet_img.show()
    
    def enhance(img_arr):
        
        for i in range(len(img_arr)):
            for w in range(len(img_arr[0])):
                if img_arr[i, w] > 66:
                    img_arr[i, w] = 0
                else: 
                    img_arr[i, w] = 255
        return img_arr
    
    def smart_divide(img_name, x, y, w, h):
        num_crop = img_crop(sheet, x, y, w, h)
        num_crop.save(img_name + '.jpg')
        
        num_arr = np.asarray(num_crop)
        enhance(num_arr)
        
        reshape_num_arr = num_arr.reshape((w, h))
        clear_line_arr = []
        global sum_line
        x = math.floor(w/2)
        try:
            for i in range(math.floor(w/2)-10, math.ceil(w/2)+10):
                sum_line = 0
                for w in range(h):
                    sum_line = sum_line + reshape_num_arr[i, w]
                    
                print(sum_line)
                if sum_line == 0:
                    clear_line_arr.append(i)
            print(clear_line_arr)
            x = clear_line_arr[math.floor(len(clear_line_arr)/2)]
        except IndexError:
            print("Attempted smart divide with image of size 0!")
        

        return x
        
        

    def find_num(img_name, x, y, w, h):
        num_crop = img_crop(sheet, x, y, w, h)
        num_crop.save(img_name + '.jpg')
        
        num_arr = img_to_arr(img_name + '.jpg', False)
        enhance(num_arr)
        
        reshape_num_arr = num_arr.reshape((img_size, img_size))
        for i in range(img_size):
            if reshape_num_arr[i].sum() == 0 and i > img_size/2:
                num_crop = img_crop(sheet, x, y-i, w, h)
            elif reshape_num_arr[i].sum() == 0 and i <= img_size/2:
                num_crop = img_crop(sheet, x, y+i, w, h)
        
        num_arr = img_to_arr(img_name + '.jpg', False)
        enhance(num_arr)
    
        image = Image.fromarray(num_arr)
        image.save(img_name + '.jpg')
        plt.imshow(num_arr)
        prediction = predict(img_name)
        print(prediction)
        return prediction
    
    def show_area(img_name, x, y, w, h):
        num_crop = img_crop(sheet, x, y, w, h)
        
        num_arr = np.asarray(num_crop)
        enhance(num_arr)
        
        for i in range(h):
            if num_arr[i].sum() == 0 and i > h/2:
                num_crop = img_crop(sheet, x, y-i, w, h)
            elif num_arr[i].sum() == 0 and i <= h/2:
                num_crop = img_crop(sheet, x, y+i, w, h)
        
        num_arr = np.asarray(num_crop)
        enhance(num_arr)
        
        image = Image.fromarray(num_arr)
        image.save(img_name + '.jpg')

        return image
        
        
    
    def pix_sum(img_name, x, y, w, h):
        num_crop = img_crop(sheet, x, y, w, h)
        num_crop.save(img_name + '.jpg')
        num_arr = np.asarray(num_crop)
        
        enhance(num_arr)
        
        for i in range(h):
            if num_arr[i].sum() == 0 and i > h/2:
                num_crop = img_crop(sheet, x, y-i, w, h)
            elif num_arr[i].sum() == 0 and i <= h/2:
                num_crop = img_crop(sheet, x, y+i, w, h)
        
        num_arr = np.asarray(num_crop)
        enhance(num_arr)
        image = Image.fromarray(num_arr)
        image.save(img_name + '.jpg')
        plt.imshow(num_arr)
        return num_arr.sum()
    
    def higher_fill(fill1, fill2, fill3, fill4, fill5):
        arr = np.array([fill1, fill2, fill3, fill4, fill5])
        min_fill_index = np.argmin(arr)
        #min_fill_val = arr[min_fill_index]
        return min_fill_index
    
    def heatmap():
        num_crop = img_crop(Image.fromarray(sheet_aligned), 0, 225, 1200, 550)
        num_crop.save('heatmap.jpg')
        heatmap_img_large = num_crop.resize((200, 100))
        points = [(73, 40), (74, 65), (88, 80), (111, 80), (128, 60), (126, 40), (111, 20), (88, 20), (73, 40)]
        draw = ImageDraw.Draw(heatmap_img_large)
        draw.line(points, width=9, fill=255, joint="curve")
        heatmap_img_small = heatmap_img_large.resize((40, 20))
        heatmap_img = heatmap_img_small.resize((300, 150))
        num_arr = np.asarray(heatmap_img)
        
        for i in range(len(num_arr)):
            for w in range(len(num_arr[0])): 
                num_arr[i, w] = 255 - num_arr[i, w]
        
        #enhance(num_arr)
        #heatmap_img.show()
        #plt.set_xlabels(' ')
        #plt.set_ylabels(' ')
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.imshow(num_arr)
        img_buf = io.BytesIO() 

        plt.savefig(img_buf, format='jpg')
        im = Image.open(img_buf)
        im.show(title="Heatmap")

        img_buf.close()

    def write_to_sheet(x, value):
        column2 = [item for item in scoutingsheet.col_values(2) if item is not None]
        y_coord = len(column2) + 1
        scoutingsheet.update_cell(y_coord, x, value)
        if y_coord / 3 == math.floor(y_coord/3):
            scoutingsheet.update_cell(y_coord, 4, 1)
        elif y_coord / 3 == math.floor(y_coord/3)+(1/3):
            scoutingsheet.update_cell(y_coord, 4, 2)
        elif y_coord / 3 == math.floor(y_coord/3)+(2/3):
            scoutingsheet.update_cell(y_coord, 4, 3)
        
        #print(scoutingsheet.cell(510, 2))
    
    
    def read_sheet():
        
        #smart_divide('smart_div', 938, 1034, 160, 70)
        
        high_div = smart_divide('smart_div', 938, 1034, 160, 70)
        print(high_div)
        num_high = show_area('num_high', 938, 1034, 160, 70)

        num_high_left = find_num('file', 938, 1034, high_div, 70)
        num_high_right = find_num('file', 938+high_div, 1034, 160-high_div, 70)
        total_high = (int(num_high_left[0])*10) + int(num_high_right[0])
        
        num_low = show_area('num_low', 1115, 1034, 160, 70)
        num_low_left = find_num('file', 1115, 1034, 80, 70)
        num_low_right = find_num('file', 1115+80, 1034, 80, 70)
        total_low = (int(num_low_left[0])*10) + int(num_low_right[0])
        
        num_missed = show_area('num_missed', 1285, 1034, 160, 70)
        num_missed_left = find_num('file', 1285, 1034, 80, 70)
        num_missed_right = find_num('file', 1285+80, 1034, 80, 70)
        total_missed = (int(num_missed_left[0])*10) + int(num_missed_right[0])
    
        taxi_fill = show_area('taxi_y_n', 555, 115, 160, 100)
        taxi_fill_y = pix_sum('file', 555, 115, 80, 100)
        taxi_fill_n = pix_sum('file', 555+80, 115, 80, 100)

        climb_a = show_area('climb_a', 1305, 875, 120, 45)
        climb_l = pix_sum('file', 1305, 875, 45, 45)
        climb_m = pix_sum('file', 1350, 875, 30, 45)
        climb_r = pix_sum('file', 1390, 875, 45, 45)
        
        endg_a = show_area('endgame_action', 590, 825, 320, 40)
        endg_c = pix_sum('file', 600, 825, 100, 40)
        endg_l = pix_sum('file', 710, 825, 80, 40)
        endg_d = pix_sum('file', 815, 825, 100, 40)
        
        cargo_a = show_area('cargo_a', 605, 780, 140, 30)
        cargo_y = pix_sum('file', 605, 780, 70, 30)
        cargo_n = pix_sum('file', 605+70, 780, 70, 30)
        
        defense_a = show_area('def_a', 885+70, 775, 140, 30)
        defense_y = pix_sum('file', 885+70, 775, 60, 30)
        defense_n = pix_sum('file', 885+140, 775, 60, 30)
        
        disable_a = show_area('dis_a', 885+400, 775, 140, 30)
        disable_y = pix_sum('file', 885+415, 775, 70, 30)
        disable_n = pix_sum('file', 885+486, 775, 60, 30)
        
        disable_y_n = higher_fill(disable_y, disable_n, 255*70*70, 255*70*70, 255*70*70)
        defense_y_n = higher_fill(defense_y, defense_n, 255*70*70, 255*70*70, 255*70*70)
        cargo_y_n = higher_fill(cargo_y, cargo_n, 255*70*70, 255*70*70, 255*70*70)
        endgame_action = higher_fill(endg_c, endg_l, endg_d, 255*80*100, 255*80*100)
        taxi_y_n = higher_fill(taxi_fill_n, taxi_fill_y, 255*80*100, 255*80*100, 255*80*100)
        climb_loc = higher_fill(climb_l, climb_m, climb_r, 255*80*100, 255*80*100)
        #heatmap()
         
        print(int(taxi_y_n))
        #rung_reached = find_num('rung_num', 488, 547, 120, 35)
    
        print(total_high)
        print(total_low)
        print(total_missed)
        
        #scoutingsheet.update_cell(510, 2, total_high)

        if climb_loc == 0:
            write_to_sheet(17, 'L')
        elif climb_loc == 1:
            write_to_sheet(17, 'M')
        elif climb_loc == 2:
            write_to_sheet(17, 'R')
            
        if endgame_action == 0:
            write_to_sheet(14, 'C')
        elif endgame_action == 1:
            write_to_sheet(14, 'L')
        elif endgame_action == 2:
            write_to_sheet(14, 'D')
        
        if cargo_y_n == 0:
            write_to_sheet(11, '1')
        elif cargo_y_n == 1:
            write_to_sheet(11, '0')
        else:
            write_to_sheet(11, 'Nothing was circled')

        if defense_y_n == 0:
            write_to_sheet(12, '1')
        elif defense_y_n == 1:
            write_to_sheet(12, '0')
        else:
            write_to_sheet(12, 'Nothing was circled')
            
        if disable_y_n == 0:
            write_to_sheet(13, '1')
        elif disable_y_n == 1:
            write_to_sheet(13, '0')
        else:
            write_to_sheet(13, 'Nothing was circled')
        
        write_to_sheet(6, int(taxi_y_n))

        write_to_sheet(18, total_high)
        write_to_sheet(19, total_low)
        write_to_sheet(20, total_missed)
        
        try:
            print(team_num)
            write_to_sheet(2, team_num)
            team_num_check = int(team_num)
        except NameError:
            print("Please enter team number!")
            Label(ws, text='Enter team number!', foreground='red').grid(row=6, columnspan=3, pady=10)
            write_to_sheet(2, "No team number was entered")
        except ValueError:
            print("Team number must be a number!")
            Label(ws, text='Team number must be a number!', foreground='red').grid(row=6, columnspan=2, pady=10)
            
        canvas = Canvas(
        ws, 
        width = 500, 
        height = 500
        )  
        
        canvas.grid(row=7,columnspan=2)
        
        img = ImageTk.PhotoImage(Image.open("num_high.jpg"))
        canvas.create_image(20,120, anchor=NW, image=img)
        
        img = ImageTk.PhotoImage(Image.open("num_low.jpg"))
        canvas.create_image(150,120, anchor=NW, image=img)
        
        img = ImageTk.PhotoImage(Image.open("num_missed.jpg"))
        canvas.create_image(280,320, anchor=NW, image=img)
        
        images = os.listdir()
        imglist = [x for x in images if x.lower().endswith(".jpg")]
    
        for index, image in enumerate(imglist): #looping through the imagelist
            photo_file = Image.open(image) 
            photo_file = photo_file.resize((150, 150),Image.ANTIALIAS) #resizing the image
            photo = ImageTk.PhotoImage(photo_file) #creating an image instance
            label = Label(image=photo)
            label.image = photo
            label.grid(row=8, column=index) #giving different column value over each iteration
            print("reached here with "+image)
        
        print("written")
        #scoutingsheet.update_cell(2, 2, total_low) 
        #scoutingsheet.update_cell(2, 3, total_missed) 

    read_sheet()

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

    


#file_selector = tk.OptionMenu(window,tk.IntVar(),for name in namelist)
#file_selector.grid(column=1,row=0)

#Define a function to close the window

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
    #progress_bar.start()
    global scouting_sheet_name
    scouting_sheet_name = str(string_path[-40:-33])
    #progress_bar.destroy()
    print(scouting_sheet_name)
    if file_path is not None:
        Label(ws, text='File Uploaded Successfully!', foreground='green').grid(row=4, columnspan=3, pady=10)
        pass
   

    
sheet_upload = Label(
    ws, 
    text=''
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
    command=run_all
    )
close.grid(row=3, columnspan=3, pady=10)

ws.mainloop()

    

