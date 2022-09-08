
import numpy as np
import cv2 as cv
import os

cascade=cv.CascadeClassifier('face.xml')


MODEL=cv.face.LBPHFaceRecognizer_create()
MODEL.read('./face_model.yml')
ROOT_DIR='./faces'
img=cv.imread('./test/gettyimages-617645216-612x612.jpg')

peopls=os.listdir(ROOT_DIR)
imgG=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=cascade.detectMultiScale(imgG,1.1,4)

for (x,y,w,h) in faces:
  imgT=imgG[y:y+h,x:x+h]
  cv.imshow('img',imgT)
  res=MODEL.predict(imgT)

  cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
  cv.putText(img,str(peopls[res[0]]),(x,y-15),fontFace=cv.FONT_ITALIC,fontScale=1.3,color=(0,0,255),thickness=3)

  cv.putText(img,f'Conf -{res[1]}',((x+w-50)//2,y+h+20),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.6,color=(0,0,0),thickness=2)

  cv.imshow('img',img)


cv.waitKey(0)
cv.destroyAllWindows()