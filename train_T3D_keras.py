# Code to train T3D model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, \
                            TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras import losses
import keras.backend as K
import traceback
import time

from T3D_keras import T3D169_Dropout
from get_video import video_gen
from adabound import AdaBound
from focal_loss import binary_focal_loss


FRAMES_PER_VIDEO = 32
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
FRAME_CHANNEL = 3
NUM_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 200
MODEL_FILE_NAME = 'T3D.h5'


def train():
    sample_input = np.empty(
        [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    # Read Dataset
    d_train = pd.read_csv(os.path.join('train.csv'))
    d_valid = pd.read_csv(os.path.join('test.csv'))
    # Split data into random training and validation sets
    nb_classes = len(set(d_train['class']))

    video_train_generator = video_gen(
        d_train, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE, augmentations=True)
    video_val_generator = video_gen(
        d_valid, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE, augmentations=False)

    # Get Model
    # model = densenet121_3D_DropOut(sample_input.shape, nb_classes)
    model = T3D169_Dropout(sample_input.shape, nb_classes, d_rate=0.2)

    checkpoint = ModelCheckpoint('T3D_weights.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    # reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                                    patience=20,
    #                                    verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
    csvLogger = CSVLogger('T3D_history.csv', append=True)
    tensorboard = TensorBoard(log_dir=f'./logs/T3D-{int(time.time())}')

    callbacks_list = [checkpoint, csvLogger, tensorboard]

    # compile model
    #optim = Adam(lr=1e-4, decay=1e-6)
    #optim = SGD(lr = 0.1, momentum=0.9, decay=1e-4, nesterov=True)
    optim = AdaBound()
    model.compile(optimizer=optim, loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])
    
    if os.path.exists('./T3D_weights.hdf5'):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights('./T3D_weights.hdf5')
        print('Weights loaded')

    # train model
    print('Training started....')

    train_steps = len(d_train)//BATCH_SIZE
    val_steps = len(d_valid)//BATCH_SIZE

    history = model.fit_generator(
        video_train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=video_val_generator,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callbacks_list,
        workers=1,
        use_multiprocessing=True
    )
    model.save(MODEL_FILE_NAME)


if __name__ == '__main__':
    try:
        train()
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()