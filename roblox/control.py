# abstraction API for Roblex keyboard and mouse control

import numpy as np
import random
import time
from ahk import AHK

# key mappings
key = {'up': 'w',
       'down': 's',
       'left': 'a',
       'right': 'd',
       'jump': 'space',
       'turn_left': 'left',
       'turn_right': 'right'}
keyTurns = [key['turn_left'], key['turn_right']]
ahk = AHK()

class Control:
    def __init__(self, simulationMode=False):
        self.actionPressed = []
        self.simulationMode = simulationMode

    def press(self, action):
        '''press the key associated with action '''
        assert action in key, 'Error invalid action %s'%action
        self.actionPressed.append(action)
        if not self.simulationMode:
            ahk.key_down(key[action], blocking=False)

    def release_all_keys(self):
        '''Release all keys pressed'''
        if not self.simulationMode:
            for a in self.actionPressed:
                ahk.key_up(key[a], blocking=False)
        self.actionPressed = []

    def all_move_actions(self):
        ''' return all move actions '''
        return ['up', 'down', 'left', 'right']

    def current_actions(self):
        '''return current actions in motion'''
        return self.actionPressed.copy()






