import time
class FrameCounter:
    def __init__(self, interval=5):
        '''
            interval: time in sec between printing out fps values.
        '''
        self.interval = interval
        self.frames = None
        self.start_time = None

    def log(self):
        '''log one frame, print out fps if needed'''
        if self.frames is None:
            self.frames = 0
            self.start_time = time.time()
            return

        # calculate fps
        self.frames += 1
        dt = time.time() - self.start_time
        if  dt > self.interval:
            self.fps = float(self.frames)/dt
            print('fps is %1.1f'%self.fps)
            self.start_time = time.time()
            self.frames = 0