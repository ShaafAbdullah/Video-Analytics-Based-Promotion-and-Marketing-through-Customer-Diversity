import glob
import cv2
from PIL import Image, ImageEnhance
path="E:/FYP37CE-B/Seperated/face_classification-master/datasets/AgeGroupFinal/SeperatedAgeGroup/kidsPerfect/*.*"
for bb,file in enumerate (glob.glob(path)):
    im = Image.open(file)
    enhancer = ImageEnhance.Contrast(im)
    enhanced_im = enhancer.enhance(2.0)
    print(enhanced_im)
    enhanced_im.save('E:/FYP37CE-B/Seperated/face_classification-master/datasets/AgeGroupFinal/SeperatedAgeGroup/Rotated/kidsrotated/kidschanged/{}.png'.format(bb))
