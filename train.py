import argparse
import os
import tensorflow as tf
import numpy as np

import pandas as pd

from glob import glob


from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.callbacks import Callback

from util.metrics import AP, get_ppn_loss
from util.network import SubParts_SSD_PPN_ResNet50
from util.batch_generator import SubPartsBatchGenerator, output_decoder, subparts_output_decoder
from util.augmentation import augmentation_seq
from util.callbacks import EvaluateMeanAP

from dvclive import Live
from dvclive.keras import DvcLiveCallback

# Parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--experiment", default="current", help="Experiment name (no space allowed)")
parser.add_argument("-e", "--epochs", default=10, type=int, help="Number of epochs to train")
args = vars(parser.parse_args())

# Setup parameters
DATASET = 'data'
CHECKPOINTS = f'/scratch/diogo.alves/checkpoints/{args["experiment"]}'
BATCH_SIZE = 8
EPOCHS = args['epochs']

# Initialization
np.random.seed(1)
if not  os.path.exists(CHECKPOINTS):
    os.makedirs(CHECKPOINTS)
if not len(tf.config.list_physical_devices('GPU')) > 0:
    print(f'ERROR: No GPU found!!!')
    exit(1)

live = Live(f'training_metrics/{args["experiment"]}',resume=True)
initial_epoch = 0

# Resume training or compile a new model
network = SubParts_SSD_PPN_ResNet50(class_labels = ['qr_code'],
                                     subparts_class_labels = ['fip'],
                                     input_shape = (480,480,3))

checkpoints = sorted(glob(f'{CHECKPOINTS}/*.tf'))

if len(checkpoints) > 0:
    model_path = checkpoints[-1]
    print(f'[Resuming training from checkpoint: {model_path}]')
    with custom_object_scope({'resnet50': tf.keras.applications.resnet50,
                        'relu6': tf.keras.layers.ReLU(6.),
                        'ppn_loss': get_ppn_loss()}):
        model = tf.keras.models.load_model(model_path)
        
        position = model_path.index('model.')
        initial_epoch = int(model_path[position+6:position+12])
        EPOCHS += initial_epoch

else:
    print('[No checkpoints found, starting a new training experiment]')
    model = network.model

network.model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-4, decay=1e-4),
    loss = {'subparts_output': get_ppn_loss(), 'main_output': get_ppn_loss()},
    loss_weights = {'subparts_output': 1.0, 'main_output': 1.0}
)
model.summary()


#### TRAIN DATA
train_qr_codes = pd.read_csv(f'{DATASET}/v2_qr_codes_train.csv', dtype={'image_id': str, 'object_id': str})
valid_qr_codes = pd.read_csv(f'{DATASET}/v2_qr_codes_valid.csv', dtype={'image_id': str, 'object_id': str})
test_qr_codes  = pd.read_csv(f'{DATASET}/v2_qr_codes_test.csv',  dtype={'image_id': str, 'object_id': str})

train_fips = pd.read_csv(f'{DATASET}/v2_fips_train.csv', dtype={'image_id': str, 'object_id': str})
valid_fips = pd.read_csv(f'{DATASET}/v2_fips_valid.csv', dtype={'image_id': str, 'object_id': str})
test_fips  = pd.read_csv(f'{DATASET}/v2_fips_test.csv',  dtype={'image_id': str, 'object_id': str})

batch_generator_train = SubPartsBatchGenerator(network)
batch_generator_valid = SubPartsBatchGenerator(network)
batch_generator_test  = SubPartsBatchGenerator(network)

batch_generator_train.add_data(dataset = train_qr_codes, subparts_dataset = train_fips, images_dir=f'{DATASET}/images')
batch_generator_valid.add_data(dataset = valid_qr_codes, subparts_dataset = valid_fips, images_dir=f'{DATASET}/images')
batch_generator_test.add_data(dataset = test_qr_codes,  subparts_dataset = test_fips,  images_dir=f'{DATASET}/images')

# Used by EvaluateValidMeanAP
valid_generator, valid_size = batch_generator_valid.get_generator(batch_size = 1, shuffle = False, augmentation = False, encode_output = False)
valid_X, valid_y_subparts, valid_y_obj = [], [], []

for _ in range(valid_size):
    batch_X, batch_y = next(valid_generator)
    valid_X.append(batch_X[0])
    valid_y_subparts.append(batch_y[0][0])
    valid_y_obj.append(batch_y[1][0])

valid_X = np.array(valid_X)

# Used by EvaluateTestMeanAP
test_generator, test_size = batch_generator_test.get_generator(batch_size = 1, shuffle = False, augmentation = False, encode_output = False)
test_X, test_y_subparts, test_y_obj = [], [], []

for _ in range(test_size):
    batch_X, batch_y = next(test_generator)
    test_X.append(batch_X[0])
    test_y_subparts.append(batch_y[0][0])
    test_y_obj.append(batch_y[1][0])

test_X = np.array(test_X)

# Used by EvaluateTrainMeanAP
train_generator, train_size = batch_generator_train.get_generator(batch_size = 1, shuffle = False, augmentation = False, encode_output = False)
train_X, train_y_subparts, train_y_obj = [], [], []

for _ in range(train_size):
    batch_X, batch_y = next(train_generator)
    train_X.append(batch_X[0])
    train_y_subparts.append(batch_y[0][0])
    train_y_obj.append(batch_y[1][0])
train_X = np.array(train_X)

# Preparing the input data
train_generator, train_size = batch_generator_train.get_generator(batch_size=BATCH_SIZE, shuffle=True, augmentation=augmentation_seq, encode_output=True)
valid_generator, valid_size = batch_generator_valid.get_generator(batch_size = BATCH_SIZE, shuffle = False, augmentation = False, encode_output = True)

print('%d training images' % train_size)
print('%d validation images' % valid_size)
print('%d test images' % test_size)

batch_X, batch_y = next(train_generator)
print(f'batch_X.shape={batch_X.shape}')
print(f'batch_y[0].shape={batch_y[0].shape}, batch_y[1].shape={batch_y[1].shape}')

network.model.fit(train_generator,
                        initial_epoch=initial_epoch,
                        steps_per_epoch = train_size // BATCH_SIZE,
                        epochs = EPOCHS,
                        validation_data = valid_generator,
                        validation_steps = valid_size // BATCH_SIZE,
                        verbose = 1,
                        max_queue_size = 200,
                        callbacks = [
                            DvcLiveCallback(path=f'training_metrics/{args["experiment"]}', resume=True),
                            ModelCheckpoint(
                                filepath=f'{CHECKPOINTS}/model.{{epoch:06d}}.tf',
                                save_freq=(train_size // BATCH_SIZE) * 10,
                                verbose=1
                            ),
                            EvaluateMeanAP(
                                network,
                                BATCH_SIZE, 
                                live,
                                train = {
                                    'X': train_X, 
                                    'y_obj': train_y_obj,
                                    'y_subparts':train_y_subparts
                                },
                                valid = {
                                    'X': valid_X, 
                                    'y_obj': valid_y_obj,
                                    'y_subparts':valid_y_subparts
                                },
                                test = {
                                    'X': test_X, 
                                    'y_obj': test_y_obj,
                                    'y_subparts':test_y_subparts
                                }
                            ),
                        ]) 