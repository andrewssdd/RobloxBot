from utils import robloxControl as rc
from games.mm2 import mm2_coinDetector
import numpy as np
import matplotlib.pyplot as plt
from games.mm2.mm2_policy import policy
import cv2
import time

mm2_coinDetector.DEBUG = True

if __name__ == '__main__':
    plt.ion()
    DEBUG_image =  r'C:\Users\ctawo\PycharmProjects\autoclicker\mm2' \
    r'\12-07-2020_23-33-06\img_0168.png'


    # coin templates
    tt = r'C:\Users\ctawo\PycharmProjects\autoclicker\mm2\dcoin%d.png'
    templates = [tt % i for i in [1,3,5]]
    img = cv2.imread(DEBUG_image)
    for i,t in enumerate(templates):
        plt.figure(1)
        hfig1 = rc.getHandleByTitle('Figure 1')
        coinDetector = mm2_coinDetector.CoinDetector_tm([t])
        x,y = coinDetector.findCoins(img)

        #print(x,y)
        plt.clf()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.plot(x,y, 'x', color='white')
        plt.plot(np.median(x), np.median(y), 'o', color='white')
        xc, yc = [0.5, 0.75]
        plt.plot(xc*img.shape[1], yc*img.shape[0], '^', color='red')
        dx, dy, direction = policy(np.array(x)/img.shape[1], np.array(y)/img.shape[0], xc, yc, simulationMode=True)
        plt.title('%s, %d, %1.2f, %1.2f, %s'%(t.split('\\')[-1], len(x), dx, dy, direction) )
        rc.moveWindow(hfig1, 0, 0, 900, 800)  # next to roblox screen
        # img3 = cv2.drawMatches(coinDetector.img_templates[0],
        #                        coinDetector.kp_template[0],
        #                        img,
        #                        coinDetector.kp_target,
        #                        coinDetector.matches[0],
        #                        img, flags=2)
        # plt.imshow(img3)
        # plt.pause(0.001)

        pass
        #input("Press [enter] to continue.")


    plt.figure(1)
    hfig1 = rc.getHandleByTitle('Figure 1')

    coinDetector = mm2_coinDetector.CoinDetector_tm(templates)
    starttime=time.time()
    x,y = coinDetector.findCoins(img)
    duration = time.time()-starttime
    #print(x,y)
    plt.clf()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(x, y, 'x', color='white')
    plt.plot(np.median(x), np.median(y), 'o', color='white')

    xc, yc = [0.5, 0.75]
    plt.plot(xc*img.shape[1], yc*img.shape[0], '^', color='red')
    dx, dy, direction = policy(np.array(x)/img.shape[1], np.array(y)/img.shape[0], xc, yc, simulationMode=True)
    plt.title('all, %d, %1.2f, %1.2f, %s, duration %1.2f'%(len(x), dx, dy, direction, duration) )
    rc.moveWindow(hfig1, 0, 0, 900, 800)  # next to roblox screen
    plt.pause(0.001)
    input("Press [enter] to continue.")
