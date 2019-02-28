import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
import random

ROOT_PATH = ''

# TODO: Add random cropping (from 256x256 to 224x224)

def get_video_frames(src, fpv=32, frame_height=224, frame_width=224):
    # print('reading video from', src)
    cap = cv2.VideoCapture(src)
    frames = []
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # set start frame number
    try:
        seq_begin = random.randint(0, num_frames-fpv)
    except ValueError:
        # If there aren't enough frames, just start at the beginning
        seq_begin = 0
        print(src, " doesn't have enough frames, padding with the last frame")

    cap.set(cv2.CAP_PROP_POS_FRAMES, seq_begin)

    if not cap.isOpened():
        cap.open(src)
    ret = True
    while(fpv and ret):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(not ret):
            break
        # Convert BGR->RGB
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        frames.append(frame)
        fpv-=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    # If captured less than fpv frames, repeat the last frame
    while fpv!=0:
        frames.append(frames[len(frames)-1])
        fpv-=1

    # print(len(frames))
    frames = [cv2.resize(f,(frame_width, frame_height)) for f in frames]
    frames = [cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for f in frames]

    return np.asarray(frames)


# Gets the videos and labels for transfer learning
def get_video_and_label_TL(path_to_videos, frames_per_video, frame_height, frame_width):
    # 0 - videos don't match; 1 - videos do match
    label = random.randint(0,1)
    video_src = os.path.join(path_to_videos, random.choice(os.listdir(path_to_videos)))
    frames = get_video_frames(video_src, frames_per_video, frame_height, frame_width)
    frames = np.expand_dims(frames, axis=0)
    if label == 1:
        return frames, frames, 1

    video_src2 = video_src
    # to avoid the same video getting picked by chance (however small)
    while(video_src2 == video_src):
        video_src2 = os.path.join(path_to_videos, random.choice(os.listdir(path_to_videos)))
    
    frames2 = get_video_frames(video_src2, frames_per_video, frame_height, frame_width)
    frames2 = np.expand_dims(frames2, axis=0)

    return frames, frames2, 0

# Video generation for transfer learning, num_classes should be 2
def video_gen(path_to_videos, frames_per_video, frame_height, frame_width, channels, num_classes=2, batch_size=4):
    while True:

        for batch in range(0, batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            frames1 = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)
            frames2 = frames1

            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get frames and its corresponding color for an traffic light
                single_frame, single_clip, sport_class = get_video_and_label(
                    i, data, frames_per_video, frame_height, frame_width)

                # Appending them to existing batch
                frame = np.append(frame, single_frame, axis=0)
                clip = np.append(clip, single_clip, axis=0)

                y_train = np.append(y_train, [sport_class])
            y_train = to_categorical(y_train, num_classes=num_classes)

            yield ([frame, clip], y_train)


