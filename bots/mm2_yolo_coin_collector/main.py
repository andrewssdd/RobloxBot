import sys
sys.path.append("../../")  # root folder
sys.path.append("../../yolo/yolov5")  # yolov5 folder, needed for pretrained model to work

from roblox import screen
import policy
import cv2
import time
import keyboard
import yolo.utils
import random
from roblox.utils import FrameCounter
from bot_utils import opt
from bot_utils import AnnotedImage

saveStream = opt.save_stream
simulationMode = opt.simulation_mode
weights = opt.weights
device = opt.device
augment = opt.augment
saveImg = opt.save_annote_img

if __name__ == '__main__':
    # load pretrained model
    device = yolo.utils.select_device(device)
    model = yolo.utils.loadModel(weights, device)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # background process to capture roblox window
    stream = screen.CaptureStream('Roblox', stride=model.stride.max(), scale=1., saveInterval = saveStream)

    # initialize
    policy.init_control(simulationMode)
    frame_counter = FrameCounter(interval=5)
    anno_image = AnnotedImage(saveImg)


    for img, img0 in stream:
        # infer coin and person positions
        pred, img1 = yolo.utils.infer(model, img, device)
        pos = yolo.utils.processPrediction(pred, img1, img, names, colors)

        # apply policy
        dx, dy, actions, msg = policy.perform_actions(pos)
        if actions != []:
            print('dx %1.2f, dy %1.2f, %s, %s'%(dx, dy, actions, msg))

        # show image
        anno_image.show(img, dx, dy, actions, pos, msg)

        # count frames
        frame_counter.log()


