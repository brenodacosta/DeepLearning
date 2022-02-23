import cv2, os

def frameGenerator(video,path):
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    path = './frames/'
    if not os.path.exists(path):
        os.makedirs(path) 
    while success:
        cv2.imwrite(path+"frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

