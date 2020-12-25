import sys
sys.path.append("../../")  # root folder
import argparse
import cv2
from roblox import screen
import policy
import string
from datetime import datetime
import os
import random
import keyboard
import threading
import matplotlib.pyplot as plt
import numpy as np
from roblox.screen import moveWindow, getHandleByTitle

parser = argparse.ArgumentParser()
parser.add_argument('--simulation-mode', action='store_true', help='Not performing policy action')
parser.add_argument('--save-stream', type =float, default=0., help='Save images from capture stream. 0 == not save. Interval to save in seconds.')
parser.add_argument('--save-annote-img', action='store_true', help='Save annoted images')
opt = parser.parse_args()
print(opt)

COIN = 0

class AnnotedImage:
    def __init__(self, save = False):

        self.save = save
        if save:
            script_path = os.path.dirname(os.path.realpath(__file__))
            randomID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            foldername = script_path + '\\images\\annoted_img_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            os.mkdir(foldername)
            self.filename = foldername + r'\\' + randomID + '_img_%04d.png'
            self.imgi = 0
        self.hfig = None
        plt.ion()

    def show(self, img, x, y, xc, yc, actions, msg):
        ''' show image

        :param img: captured image
        :param x: relative x coordinates of coins
        :param y: relative x coordinates of coins
        :param actions: list of actions performed
        :param pos: positions of coins
        :param msg: message to display on image
        :return:
        '''
        for x1,y1 in zip(x,y):
            img = cv2.circle(img, (int(x1), int(y1)), 10, [0,255,0], 2)
        img = cv2.circle(img, (int(xc*img.shape[1]), int(yc*img.shape[0]) ), 10, [255, 255, 0], 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('mm2_stream', img)
        if self.hfig is None:
            self.hfig = screen.getHandleByTitle('mm2_stream')
            screen.moveWindow(self.hfig, 1000, 0)
        if cv2.waitKey(1) == ord('q') or keyboard.is_pressed('q'):  # q to quit
            print('Exit bot.')
            sys.exit(0)
        if self.save:
            # new thread to save time
            thread = threading.Thread(target=cv2.imwrite, args=(self.filename%self.imgi, img))
            thread.start()
            #cv2.imwrite(self.filename%self.imgi, img)
            self.imgi += 1


        pass

