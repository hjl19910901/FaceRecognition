import cv2  # opencv库

# 读取图片
image = cv2.imread('image/2.jpg')
#加载人脸模型库
face_model=cv2.CascadeClassifier('plugins/opencv/haarcascades/haarcascade_frontalcatface.xml')
#图片进行灰度处理
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#人脸检测
faces = face_model.detectMultiScale(gray)
#标记人脸
for (x,y,w,h) in faces:
    #1.原始图片；2坐标点；3.矩形宽高 4.颜色值(RGB)；5.线框
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
# 显示图片窗口
cv2.imshow('faces', image)
# 窗口暂停
cv2.waitKey(0)
# 销毁窗口资源
cv2.destoryAllWindows()
