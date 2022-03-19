import cv2
import numpy as np

def im_read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def im_write(path, image, format=".png"):
    cv2.imencode(format, image)[1].tofile(path)

# Usage

# image = im_read("D:/Project/CJYM/sparepart_detection/test/01垫片/2B9-H4328-01-00-81/20210916200339745_catch.png")
# print(image.shape)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# im_write("D:/Project/CJYM/sparepart_detection/test/临时保存/tmp.png", image)
