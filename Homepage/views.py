import os

import cv2
from django.core.files.storage import FileSystemStorage, default_storage
from matplotlib import pyplot as plt
import numpy as np
import imutils
import numpy as np
import cv2
import tensorflow as tf
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.



def home(request):
    return render(request, 'a.html', {'titles': 'Django', 'link': 'http://127.0.0.1:8000'})


def profile(request):
    return render(request, 'a.html', {'titles': ', this is profile page', 'link': 'http://127.0.0.1:8000/'})


def expression(request):
    if request.method == 'POST':
        img_name = request.FILES['car_image']

        fs = FileSystemStorage()
        filePathName = fs.save(img_name.name, img_name)

        f = fs.url(filePathName)

        f = f[1:]
        char = plate_detection(f)
        show_results(char)

        g = fs.url('read2.jpg')
        g = g[1:]

        img_list = [g]

        context = {'images': img_list}

        return render(request, 'a.html', context)


def plate_detection(input):
    image = cv2.imread(input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 170, 200)
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img1 = image.copy()
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None
    count = 0
    idx = 7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)  # output number of edge in contour
        if len(approx) == 4:  # contour with 4 corners
            location = approx  # our approx numnuber plate contour

            # crop contours and store it into cropped images folder
            x, y, w, h = cv2.boundingRect(c)  # find co-ordinate for plate
            new_img = image[y:y + h, x:x + w]  # create new image
            cv2.imwrite('Cropped Images-Text/' + str(idx) + '.png', new_img)  # save new image
            idx += 1
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    cv2.imwrite('media/plate.jpg', cropped_image)
    img = cv2.imread('media/plate.jpg')
    char = segmentation(img)
    print('Love')
    return char


def contouring(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bottom_width = dimensions[0]
    top_width = dimensions[1]
    bottom_height = dimensions[2]
    top_height = dimensions[3]

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    read_2 = cv2.imread('media/contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        if intWidth > bottom_width and intWidth < top_width and intHeight > bottom_height and intHeight < top_height:
            x_cntr_list.append(intX)
            char_copy = np.zeros((44, 24))

            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(read_2, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(read_2, cmap='gray')
            cv2.imwrite('media/read2.jpg', read_2)
            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res


def segmentation(image):
    segment_image = cv2.resize(image, (333, 75))
    segment_image_gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    _, segment_image_binary = cv2.threshold(segment_image_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segment_image_binary = cv2.erode(segment_image_binary, (3, 3))
    segment_image_binary = cv2.dilate(segment_image_binary, (3, 3))

    LP_WIDTH = segment_image_binary.shape[0]
    LP_HEIGHT = segment_image_binary.shape[1]

    segment_image_binary[0:3, :] = 255
    segment_image_binary[:, 0:3] = 255
    segment_image_binary[72:75, :] = 255
    segment_image_binary[:, 330:333] = 255

    dimensions = [LP_WIDTH / 6,
                  LP_WIDTH / 2,
                  LP_HEIGHT / 10,
                  2 * LP_HEIGHT / 3]
    cv2.imwrite('media/contour.jpg', segment_image_binary)

    char_list = contouring(dimensions, segment_image_binary)

    return char_list


def fix_dimension(img='media/plate.jpg'):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    model = tf.keras.models.load_model('model/')
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):
        img_ = cv2.resize(ch, (28, 28))
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_ = model.predict(img)[0]
        character = dic[np.argmax([y_])]
        output.append(character)

    plate_number = ''.join(output)
    return plate_number
