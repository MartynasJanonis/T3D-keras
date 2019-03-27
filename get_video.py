import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
import random

ROOT_PATH = ''

# TODO: Add random cropping (from 256x256 to 224x224)
# TODO: Figure out whether it's worth mean subtracting images

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
        # print(src, " doesn't have enough frames, padding with the last frame")

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

    return frames


# Gets the videos and labels for transfer learning
def get_video_and_label_TL(path_to_videos, frames_per_video, frame_height, frame_width):
    # 0 - videos don't match; 1 - videos do match
    label = random.randint(0,1)
    video_src = os.path.join(path_to_videos, random.choice(os.listdir(path_to_videos)))
    frames = get_video_frames(video_src, frames_per_video, frame_height, frame_width)

    if label == 1:
        return frames, frames, 1

    video_src2 = video_src
    # to avoid the same video getting picked by chance (however small)
    while(video_src2 == video_src):
        video_src2 = os.path.join(path_to_videos, random.choice(os.listdir(path_to_videos)))
    
    frames2 = get_video_frames(video_src2, frames_per_video, frame_height, frame_width)

    return frames, frames2, 0


# Video generation for transfer learning, num_classes should be 2
def video_gen_TL(path_to_videos, frames_per_video, frame_height, frame_width, channels, num_classes=2, batch_size=4):
    while True:
        input_2d_batch = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)
        input_3d = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)

        y_train = np.empty([0], dtype=np.int32)

        for batch in range(batch_size):
            # get frames and the label (whether the frames match or not)
            f1, f2, label = get_video_and_label_TL(path_to_videos, frames_per_video, frame_height, frame_width)
            
            # whether to apply augmentations
            aug = random.randint(0,1)
            if aug:
                f1 = [cv2.flip(f,1) for f in f1]
                f2 = [cv2.flip(f,1) for f in f2]
                # print("AUG applied")
            f1 = np.asarray(f1)
            f2 = np.asarray(f2)
            # Normalize the videos
            f1 = (f1 - f1.min())/np.ptp(f1)
            f2 = (f2 - f2.min())/np.ptp(f2)
            f1 = np.expand_dims(f1, axis=0)
            f2 = np.expand_dims(f2, axis=0)
            # Appending them to existing batch
            input_2d_batch = np.append(input_2d_batch, f1, axis=0)
            input_3d = np.append(input_3d, f2, axis=0)

            y_train = np.append(y_train, [label])
        y_train = to_categorical(y_train, num_classes=num_classes)

        yield ([input_2d_batch, input_3d], y_train)


def get_video_and_label(index, data, frames_per_video, frame_height, frame_width):
    # Read clip and appropiately send the sports' class
    frames = get_video_frames(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()), frames_per_video, frame_height, frame_width)
    action_class = data['class'].values[index]

    frames = np.expand_dims(frame, axis=0)
    
    return frames, action_class


def video_gen(data, frames_per_video, frame_height, frame_width, channels, num_classes, batch_size=4):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            video_clips = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)

            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get frames and its corresponding color for an traffic light
                frames, action_class = get_video_and_label(
                    i, data, frames_per_video, frame_height, frame_width)

                frames = np.asarray(frames)
                # standardize the frames
                frames = (frames - np.mean(frames)) / np.std(frames)
                # Appending them to existing batch
                video_clips = np.append(video_clips, frames, axis=0)

                y_train = np.append(y_train, [action_class])
            y_train = to_categorical(y_train, num_classes=num_classes)
        
        yield (video_clips, y_train)