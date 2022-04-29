from genericpath import exists
from ntpath import join
import cv2, os, re, glob,imageio
from pathlib import Path
from cv2 import imread
import numpy as np

def frameFromVideo(video,path='./frames/'):
    # Receives a video and split it into frames
    # Frames are saved in the path (./frames/) by default
    
    # Also returns video shape  (width and length). This information can be used to crop other images to insert new frames in the video
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    video_width = vidcap.get(3) 
    video_height = vidcap.get(4)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(video_fps)
    if os.path.exists(path): # Remove old frames in the path, it they exist
        files = glob.glob(os.path.join(path,'*'))
        for f in files:
            os.remove(f)
    else:
        os.makedirs(path) 
    while success: # Read frame by frame and save it in the path
        cv2.imwrite(path+"frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    return int(video_height),int(video_width),video_fps

def frameNameModify(path,position,qty):
        # To insert a new frame in a specific order, the subsequent frames must be shifted by the number of frames to be inserted
        # Eg: ...,frame3,frame4,frame5,frame6,...,frameN
        # A new frame will be inserted after frame5 and before frame6. This function does:
        # ...,frame3,frame4,frame5,_,frame6+1,...,frameN+1
        
        dirFiles = os.listdir(path) # List frames
        dirFiles.sort(key=lambda f: int(re.sub('\D', '', f))) # To sort files numerically (1,2,3...,11,12 rather than 1,11,12,2,3)
        for index, file in reversed(list(enumerate(dirFiles))):
            if index >= position:
                os.rename(os.path.join(path,file),os.path.join(path,''.join(['frame',str(index+qty),'.jpg']))) 
        return

def cropresizeframe(frame,height_video,width_video):
    # To insert a frame in a video, it must have the same shape as the video
    # This function reshape an image by rescaling it to fill the video shape and crop the remanescent edge

    assert type(frame) == np.ndarray, 'height_video must be int'
    assert type(height_video) == int, 'height_video must be int'
    assert type(width_video) == int, 'width_video must be int'

    height_rel = height_video/frame.shape[0] # Relationship between video height and the height of the image to be inserted
    width_rel = width_video/frame.shape[1] # Relationship between video width and the width of the image to be inserted

    scale_reshape = height_rel if height_rel > width_rel else width_rel # It is considered the biggest difference between dimensions, so the frame will cover the whole video shape (and a part of it will be cropped)

    new_height = int(frame.shape[0]*scale_reshape)
    new_width = int(frame.shape[1]*scale_reshape)

    resized_image = cv2.resize(frame, (new_width, new_height))

    # The cropped image is placed in the center of the original image
    crop_img= resized_image[int((new_height-height_video)/2):int((new_height-height_video)/2)+height_video,int((new_width-width_video)/2):int((new_width-width_video)/2)+width_video]

    return crop_img

def videoFromFrameFolder(path,framerate,video_name='untitled.mp4'):
    # Receives the path of a folder and concatenate all present images in a mp4 video
    img_array = []
    dirFiles = os.listdir(path) # List frames
    dirFiles.sort(key=lambda f: int(re.sub('\D', '', f))) # To sort files numerically (1,2,3...,11,12 rather than 1,11,12,2,3)
    
    # Create an array with the images found by listdir
    for filename in dirFiles:
        img = cv2.imread(path+filename)
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)
    
    # Create a constructor with the video parameters (name, codec, framerate, shape)
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), framerate, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    return

def gifFromFrameFolder(path,framerate=15,gif_name='untitled.gif'):
    # Receives a folder path and concatenates all present images in a gif
    filenames = os.listdir(path) # List frames
    filenames.sort(key=lambda f: int(re.sub('\D', '', f))) # To sort files numerically (1,2,3...,11,12 rather than 1,11,12,2,3)
    images = []
    for filename in filenames:
        images.append(imageio.imread(os.path.join(path,filename)))
    imageio.mimsave(gif_name, images,fps=framerate)

def frameInsert(frame_src,video_dst,position,gif=False,output_name='untitled',framerate=None):
    # Insert a frame, or an array of frames, in a video, returns the video with the added frames (or a gif if gif=True)
    
    if type(frame_src) == str:
        # check if the path exists
        assert Path(frame_src).exists(), 'File not found in '+frame_src
    elif type(frame_src) == list:
        # check if each path exists
        for f in frame_src:
            assert type(f) == str, 'File not found because the path passed is not a string'
            assert Path(f).exists(), 'File not found in '+f
    else:
        # assert type(frame_src) == np.ndarray or type(frame_src) == , 'frame_str must be str, array of str or numpy.ndarray (image read with cv2.imread())'
        assert frame_src == np.ndarray, 'If the file is not given by its location, it must be passed by argument using cv2.imread(filepath)'
    assert type(position) == int, 'frame position must be int'
    
    if framerate == None:
        height_video, width_video,framerate = frameFromVideo(video_dst) # Separate the video in frames and get shape and fdp information (if fps not specified)
    else:
        height_video, width_video,_ = frameFromVideo(video_dst) # Separate the video in frames and get shape information
    path = './frames/' # To create a folder where store the temporary frames
    
    frameNameModify(path,position,len(frame_src)) # Frames are named frame1.jpg,...,frameN.jpg, this function shifts each frame name departing from the position given by parameter

    for index,fr in enumerate(frame_src):
        if type(fr) == str:
            cropped_frame = cropresizeframe(cv2.imread(fr),height_video,width_video) # If the frame does not have the same shape of the video, this frame is resized and cropped to be in the same shape as the video
        else:
            cropped_frame = cropresizeframe(fr,height_video,width_video) # If the frame does not have the same shape of the video, this frame is resized and cropped to be in the same shape as the video
        cv2.imwrite(os.path.join(path,''.join(['frame',str(position+index),'.jpg'])),cropped_frame) # Save the frame(s) in the temp folder with the name frame(position).jpg
    if gif:
        print('llego aqui')
        output_name = ''.join([output_name,'.gif'])
        return gifFromFrameFolder(path,framerate,output_name) # Create a gif from the set of frames present in the temp folder
    else:
        output_name = ''.join([output_name,'.mp4'])
        return videoFromFrameFolder(path,framerate,output_name) # Create a video from the set of frames present in the temp folder