# Code to train T3D model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, \
                            TensorBoard, LearningRateScheduler
from keras.optimizers import Adam, SGD, Nadam
from keras import losses
import keras.backend as K
import traceback

from T3D_keras import T3D169_DenseNet
from get_video import video_gen_TL

# there is a minimum number of frames that the network must have, values below 10 gives -- ValueError: Negative dimension size caused by subtracting 3 from 2 for 'conv3d_7/convolution'
# paper uses 224x224, but in that case also the above error occurs
FRAMES_PER_VIDEO = 32
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
FRAME_CHANNEL = 3
NUM_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 200
MODEL_FILE_NAME = 'T3D_saved_model.h5'
PATH_TO_VIDEOS = '../dataset/transfer_learning/resized'


def transfer_learning():
    sample_input = np.empty(
        [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    # # Read Dataset
    # d_train = pd.read_csv(os.path.join('train.csv'))
    # d_valid = pd.read_csv(os.path.join('test.csv'))
    # # Split data into random training and validation sets
    # nb_classes = len(set(d_train['class']))
    # For transfer learning, nb_classes has to be 2
    nb_classes = 2

    video_generator = video_gen_TL(
        PATH_TO_VIDEOS, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)

    # Get Model
    # model = densenet121_3D_DropOut(sample_input.shape, nb_classes)
    model = T3D169_DenseNet(sample_input.shape, nb_classes)

    checkpoint = ModelCheckpoint('T3D_saved_model_weights.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=100)
    lrscheduler = LearningRateScheduler(lambda epoch: 0.1 * pow(10,-(epoch//30)), verbose=1)
    csvLogger = CSVLogger('history.csv', append=True)
    tensorboard = TensorBoard(log_dir='./logs/T3D_Transfer_Learning')

    callbacks_list = [checkpoint, lrscheduler, earlyStop, csvLogger, tensorboard]

    # compile model
    #optim = Adam(lr=1e-4, decay=1e-6)
    optim = SGD(lr = 0.1, momentum=0.9, decay=1e-4, nesterov=True)
    # optim = Nadam(lr=1e-4)
    model.compile(optimizer=optim, loss=['binary_crossentropy'], metrics=['accuracy'])
    
    if os.path.exists('./T3D_saved_model_weights.hdf5'):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights('./T3D_saved_model_weights.hdf5')
        print('Weights loaded')

    # train model
    print('Training started....')

    # Arbitrary numbers as the dataset is huge (many video combinations possible)
    train_steps = 1600
    val_steps = 400

    history = model.fit_generator(
        video_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=video_generator,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callbacks_list,
        workers=1,
        use_multiprocessing=True
    )
    model.save(MODEL_FILE_NAME)


if __name__ == '__main__':
    try:
        transfer_learning()
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()
