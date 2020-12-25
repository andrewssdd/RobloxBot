import numpy as np
import keyboard
import random
import time
from ahk import AHK
#import pyautogui
whentopress=0
ahk = AHK()

key = dict()
key['up'] = 'w'
key['down'] = 's'
key['left'] = 'a'
key['right'] = 'd'
keyjump = 'space'
keyTurns = ['left', 'right']

keyPressed_async = []
randomInProgress = False
randomStartTime = -1000
actionStartTime = -1000

def policy_async(x, y, xc, yc, simulationMode = False):
    ''' Asychroneous policy action '''
    global keyPressed_async
    global randomStartTime
    global actionStartTime

    minKP = 1
    jumpProb = 0.1
    actionDuration = .5  # the duration of each action is 0 to actionDuration
    randomActionDuration = 1.  # the duration of each action is 0 to actionDuration
    randomPro = 0.1
    minActionDuration = 0.25

    dx = 0
    dy = 0

    # policy
    if len(x) >= minKP and random.random() > randomPro and time.time() - actionStartTime > minActionDuration:
        # remove pressed keys
        if keyPressed_async and not simulationMode:
            for d in keyPressed_async:
                ahk.key_up(d, blocking=False)
                # pyautogui.keyUp(d)
        keyPressed_async = []

        xm = np.median(x)
        ym = np.median(y)
        dx = xm - xc
        dy = ym - yc
        if dx >0.02:
            direction = 'right'
            keyPressed_async.append(key[direction])
        if dx <-0.02:
            direction = 'left'
            keyPressed_async.append(key[direction])
        if dy > 0.02:
            direction = 'down'
            keyPressed_async.append(key[direction])
        if dy < -0.02:
            direction = 'up'
            keyPressed_async.append(key[direction])
        if random.random() < jumpProb:
            keyPressed_async.append(keyjump)
        if not simulationMode:
            for d in keyPressed_async:
                ahk.key_down(d, blocking=False)
        actionStartTime = time.time()
    else:
        # random
        if time.time() - randomStartTime > randomActionDuration:
            # remove pressed keys
            if keyPressed_async and not simulationMode:
                for d in keyPressed_async:
                    ahk.key_up(d, blocking=False)
                    # pyautogui.keyUp(d)
            keyPressed_async = []

            d = random.choice([key[k] for k in key])
            #turn = random.choice([k for k in keyTurns])
            turn = 'w'
            keyPressed_async = [d]
            if not simulationMode:
                for d in keyPressed_async:
                    #pyautogui.keyDown(d)
                    ahk.key_down(d, blocking=False)
                    #time.sleep(randomActionDuration)
            randomStartTime = time.time()

    return dx, dy, keyPressed_async


def policy(x,y, xc, yc, simulationMode = False):
    '''action policy
        x,y: list of detected coin coordinates (normalized)
        xc,yc: coordinate of player (normalized)
    '''
    minKP = 3
    jumpProb = 0.1
    actionDuration = .5  # the duration of each action is 0 to actionDuration
    randomActionDuration = 1.  # the duration of each action is 0 to actionDuration
    randomPro = 0.1
    xm = np.median(x)
    ym = np.median(y)
    dx = xm - xc
    dy = ym - yc

    keyPressed = []

    if len(x) >= minKP and random.random() > randomPro:
        if dx >0.02:
            direction = 'right'
            keyPressed.append(key[direction])
        if dx <-0.02:
            direction = 'left'
            keyPressed.append(key[direction])
        if dy > 0.02:
            direction = 'down'
            keyPressed.append(key[direction])
        if dy < -0.02:
            direction = 'up'
            keyPressed.append(key[direction])
        if random.random() < jumpProb:
            keyPressed.append(keyjump)
        if not simulationMode:
            for d in keyPressed:
                ahk.key_down(d)
            time.sleep(random.random()*actionDuration)
            for d in keyPressed:
                ahk.key_up(d)

    else:
        # random
        d = random.choice([k for k in key])
        turn = random.choice([k for k in keyTurns])
        keyPressed = [d, keyjump, turn]
        if not simulationMode:
            for d in keyPressed:
                ahk.key_down(d)
            time.sleep(randomActionDuration)
            for d in keyPressed:
                ahk.key_up(d)


    return dx, dy, keyPressed

