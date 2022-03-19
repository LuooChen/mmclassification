import im_read_write
import numpy as np


def mosiac(src_image_path, dst_image_path, new_image_path=None):
    src_image = im_read_write.im_read(src_image_path)
    dst_image = im_read_write.im_read(dst_image_path)

    h_src, w_src, c = src_image.shape
    h_dst, w_dst, c = dst_image.shape

    new_image = np.zeros([max(h_src, h_dst), w_src+w_dst, c], dtype=np.uint8)
    new_image[:h_src, :w_src, :] = src_image
    new_image[:h_dst, w_src:w_src+w_dst, :] = dst_image

    im_read_write.im_write(new_image_path, new_image, ".jpg")


src_image_path = ""
dst_image_path = ""
new_image_path = ""
mosiac(src_image_path, dst_image_path, new_image_path)
