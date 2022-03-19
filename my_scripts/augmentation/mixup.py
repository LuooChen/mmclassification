import im_read_write
import numpy as np

def mixup(src_image_path, dst_image_path, new_image_path=None):
    src_image = im_read_write.im_read(src_image_path)
    dst_image = im_read_write.im_read(dst_image_path)

    h_src, w_src, c = src_image.shape
    h_dst, w_dst, c = dst_image.shape

    new_image = np.zeros([max(h_src, h_dst), max(w_src, w_dst), c], dtype=np.uint8)
    new_image[:h_src, :w_src, :] = src_image
    cropped = new_image[:h_dst, :w_dst]
    mask = np.ones_like(dst_image, np.float32)
    mask = mask * 0.5
    new_image[:h_dst, :w_dst, :] = mask * cropped + (1 - mask) * dst_image

    im_read_write.im_write(new_image_path, new_image, ".jpg")

# Usage
src_image_path = ""
dst_image_path = ""
new_image_path = ""
mixup(src_image_path, dst_image_path, new_image_path)
