import numpy as np
import random
import time
from roblox.control import Control

rbx_control = None
COIN = 0
PERSON = 1

randomInProgress = False
randomStartTime = -1000
actionStartTime = -1000

xc, yc = [0.5, 0.75] # position of character


def init_control(simulationMode = False):
    global rbx_control
    rbx_control = Control(simulationMode)

def perform_actions(objects):
    ''' Asychroneous policy action '''
    global randomStartTime
    global actionStartTime

    jumpProb = 0.1 # probability to spam jump in a policy action
    randomActionDuration = 1.  # the duration of each action is 0 to actionDuration
    randomPro = 0.1  # probability the perform a random motion
    actionDuration = .25  # min duration for an action in sec
    turnProb = 0.3 # probability to turn when in random motion
    runAwayProb = 0. # Probability to run away from people
    dx = 0
    dy = 0
    actions = []
    msg = ''
    numCoinsDetected = len(objects[COIN])
    numPersionDetected = len(objects[PERSON])

    # check if previous action is still in process
    if time.time() - actionStartTime <= actionDuration:
        return dx, dy, actions, msg

    # policy
    if (numCoinsDetected > 0 or numPersionDetected > 0) and random.random() > randomPro:
        # Perform policy action

        # If see a person, move away
        if len(objects[PERSON]) > 0 and random.random() < runAwayProb:
            # run away from people, they can be murders!
            xm = np.mean([xy[0] for xy in objects[PERSON]])
            ym = np.mean([xy[1] for xy in objects[PERSON]])
            rbx_control.release_all_keys()
            actions = move_away_from(xm, ym, jumpProb)
            actionStartTime = time.time()
            msg = 'run away from people (%1.2f, %1.2f)'%(xm,ym)
            return dx, dy, actions, msg

        elif len(objects[COIN]) > 0:
            # collect coins
            x, y = random.choice(objects[COIN])
            rbx_control.release_all_keys()
            actions = move_to(x, y, jumpProb)
            actionStartTime = time.time()
            msg = 'collect coin at (%1.2f, %1.2f)' % (x, y)
            return dx, dy, actions, msg

    # random motion
    if time.time() - randomStartTime > randomActionDuration:
        rbx_control.release_all_keys()
        action = random.choice(rbx_control.all_move_actions())
        rbx_control.press(action)
        actions.append(action)
        if random.random() < jumpProb:
            action = 'jump'
            rbx_control.press(action)
            actions.append(action)
        if random.random() < turnProb:
            action = random.choice(['turn_left', 'turn_right'])
            rbx_control.press(action)
            actions.append(action)

        randomStartTime = time.time()
        msg = 'random motion'

    return dx, dy, actions, msg


def move_to(x, y, jumpProb = 0.1):
    '''move to object
    Args:
        x (float): x coordinate
        y (float): y coordinate
        jumpProb (float): probability to add spam jump
    '''
    actions = []
    dx = x - xc
    dy = y - yc
    if dx > 0.02:
        action = 'right'
        rbx_control.press(action)
        actions.append(action)
    if dx < -0.02:
        action = 'left'
        rbx_control.press(action)
        actions.append(action)
    if dy > 0.02:
        action = 'down'
        rbx_control.press(action)
        actions.append(action)
    if dy < -0.02:
        action = 'up'
        rbx_control.press(action)
        actions.append(action)

    if random.random() < jumpProb:
        action = 'jump'
        rbx_control.press(action)
        actions.append(action)

    return actions

def move_away_from(x, y, jumpProb = 0.1):
    '''move away from an object
    Args:
        x (float): x coordinate of the object
        y (float): y coordinate of the object
        jumpProb (float): probability to add spam jump
    '''
    actions = []
    dx = x - xc
    dy = y - yc
    if dx < 0.0:
        action = 'right'
        rbx_control.press(action)
        actions.append(action)
    if dx > 0.:
        action = 'left'
        rbx_control.press(action)
        actions.append(action)
    if dy < 0.:
        action = 'down'
        rbx_control.press(action)
        actions.append(action)
    if dy > 0.:
        action = 'up'
        rbx_control.press(action)
        actions.append(action)

    if random.random() < jumpProb:
        action = 'jump'
        rbx_control.press(action)
        actions.append(action)

    return actions