from io import BytesIO
import requests
import base64
import numpy as np
from PIL import Image
import cv2

def imageFileToBase64(filePath):
    """
    传入文件路径，得到图片的base64字符串
    :param filePath: 文件目录
    :return:
    """
    imageStr = ""
    with open(filePath, "rb") as f:
        base64_data = base64.b64encode(f.read())
        str = base64_data.decode()
        imageStr = imageStr + str
    return imageStr

def base64ToImage(base64_image):
    data = base64_image.split(',')[1]
    # print(base64_image)
    image_data = base64.b64decode(data)
    return image_data

def getApiJson(imageUrl, apiUrl):
    image_file_base64 = imageFileToBase64(imageUrl)
    # 模拟post请求
    data_search = {
        "api_key": "YMSLX_API_KEY",
        "img_result_type": "single",
        "image_file_base64": 'data:image/jpeg;base64,'+image_file_base64
    }
    response = requests.post(url=apiUrl,data=data_search)
    return response.json()

def get_mixup_image(image_url, url):
    resJson = getApiJson(image_url, url)
    data = resJson['data']
    base64_image = data['result_b64'][0]
    image_byte = base64ToImage(base64_image)
    image_data = BytesIO(image_byte)
    image = Image.open(image_data)
    nparray = np.array(image)
    b, g, r, a = cv2.split(nparray)
    b = cv2.bitwise_and(b, a)
    g = cv2.bitwise_and(g, a)
    r = cv2.bitwise_and(r, a)
    new_image = cv2.merge([b,g,r])
    return new_image

def im_write(path, image, format=".png"):
    cv2.imencode(format, image)[1].tofile(path)
