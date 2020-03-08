import cv2
import glob
import os
import shutil
from PIL import Image
import scipy
#from scipy.misc import imsave
import numpy
import math

'''
categories = glob.glob("data/cifar100/test/*")
print(categories)

for category in categories:
    print(os.path.basename(category))
    category_imgs = glob.glob(category + "/*.png")
    if "apple" in category:
        for img in category_imgs:
            if "004" in img:
                cv2_im = cv2.imread(img)
                print(cv2_im.shape)
                image = cv2.copyMakeBorder(cv2_im, 40, 40, 40, 40, cv2.BORDER_CONSTANT, 0)
                print(image.shape)
                print(category + "/paddingtest.png")
                cv2.imwrite(category + "/paddingtest.png", image)

                binarize_image(img, category + "/binaryimage.png")
        with open(category + "/test.txt", "w+") as f:
            for img in category_imgs:
                if "paddingtest" in img:
                    f.write(os.path.basename(img) + "\n")
'''

categories = glob.glob("data/101_ObjectCategories/*")

category_simple = []
for category in categories:
    category_imgs = glob.glob(category + "/*.png") + glob.glob(category + "/*.jpg")
    with open(category + "/test.txt", "w+") as f:
        for img in category_imgs:
            pil_img = Image.open(img)
            w, h = pil_img.size
            if (w < 112 or h < 112):
                print(w, h, img)
                smallest_dim = min(w, h)
                scale_factor = 112.0 / smallest_dim
                new_w = math.ceil(scale_factor * w)
                new_h = math.ceil(scale_factor * h)
                print("Scaling factor", scale_factor, "New size", (new_w, new_h))
                pil_img = pil_img.resize((new_w, new_h))
                pil_img.save(img)
            f.write(os.path.basename(img) + "\n")
    category_simple.append(os.path.basename(category))

print("Categories:")
print(categories)