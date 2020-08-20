import glob
import cv2
from PIL import Image, ImageEnhance
path="D:/FYP/dataset/01/*.*"
for bb,file in enumerate (glob.glob(path)):
    im = Image.open(file)
    enhancer = ImageEnhance.Sharpness(im)
    enhanced_im = enhancer.enhance(10.0)
    print(enhanced_im)
    enhanced_im.save('D:/FYP/dataset/03/{}.png'.format(bb))
