import im_read_write

def yoco(src_image_path, new_image_path1=None, new_image_path2=None):
    src_image = im_read_write.im_read(src_image_path)

    h_src, w_src, c = src_image.shape

    new_image1 = src_image[:int(h_src/2), :w_src, :]
    new_image2 = src_image[int(h_src/2):h_src, :w_src, :]

    im_read_write.im_write(new_image_path1, new_image1, ".jpg")
    im_read_write.im_write(new_image_path2, new_image2, ".jpg")

# Usage
src_image_path = ""
new_image_path1 = ""
new_image_path2 = ""
yoco(src_image_path, new_image_path1, new_image_path2)
