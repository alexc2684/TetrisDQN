import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 50, 'left': 0, 'width': 800, 'height': 800}

sct = mss()

while 1:
    sct.grab(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    # cv2.imshow('test', np.array(img))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
