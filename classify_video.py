import os
import sys
import cv2
import numpy as np
import keras
import keras.backend as K
import traceback


def get_video(src, frame_height=224, frame_width=224):
    # print('reading video from', src)
    cap = cv2.VideoCapture(src)
    frames = []

    if not cap.isOpened():
        cap.open(src)
    ret = True
    while(ret):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(not ret):
            break
        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()

    # print(len(frames))
    frames = [cv2.resize(f,(frame_width, frame_height)) for f in frames]
    frames = [cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for f in frames]

    return frames

def split_video_into_frames(src, fpv=32):
    frames = get_video(src)
    split_frames = []
    i = 0
    j = fpv
    for it in range(len(frames)//fpv):
        split_frames.append(frames[i:j])
        i = j
        j += fpv
    if i!= len(frames):
        remainder = frames[i:]
        while len(remainder) != fpv:
            remainder.append(remainder[-1])
        split_frames.append(remainder)

    return split_frames

def classify_with_model(src, model):
    split_video = split_video_into_frames(src, fpv=32)
    predictions = []
    for batch in split_video:
        batch = np.asarray(batch)
        # standardize the frames
        batch = (batch - np.mean(batch)) / np.std(batch)
        batch = np.expand_dims(batch, axis=0)
        predictions.append(model.predict(batch, verbose=1))

    return predictions
    

def classify_live(model, fpv=32):
    # Select which webcam to use
    cap = cv2.VideoCapture(2)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('output.avi',fourcc,24,(224,224))
    frames = []
    print("Starting capture")
    for i in range(0,fpv):
        ret, frame = cap.read()
        print("Recorded a frame")
        if(not ret):
            break
        frames.append(frame)
    cap.release()
    frames = [cv2.resize(f,(224, 224)) for f in frames]
    for f in frames:
        out.write(f)
    out.release()
    frames = np.asarray(frames)
    frames = (frames - np.mean(frames)) / np.std(frames)
    frames = np.expand_dims(frames, axis=0)
    return model.predict(frames,verbose=1)


def live_video_gen(fpv=32, w=224, h=224):
    cap = cv2.VideoCapture(2)
    frames = []
    print("Starting capture")
    while True:
        frames = []
        for i in range(0,fpv):
            ret, frame = cap.read()
            print("Recorded a frame")
            if(not ret):
                break
            frames.append(frame)
        frames = [cv2.resize(f,(w, h)) for f in frames]
        frames = np.asarray(frames)
        frames = (frames - np.mean(frames)) / np.std(frames)
        frames = np.expand_dims(frames, axis=0)
        yield frames
    cap.release()