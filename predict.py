import tensorflow as tf
import numpy as np

import pandas as pd

from glob import glob
import cv2 as cv


from tensorflow.keras.utils import custom_object_scope

from dvclive import Live
live = Live('evaluation')

np.random.seed(1)

RESULTS = '/scratch/diogo.alves/results/more-training'
batch_size = 8

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('\nSetup complete!')


from util.batch_generator import SubPartsBatchGenerator, subparts_output_decoder, output_decoder
from util.metrics import AP, get_ppn_loss, recall, false_positives
from util.network import SubParts_SSD_PPN_ResNet50


# model_path = sorted(glob(f'{RESULTS}/resnet50*'))[-1]
model_path = sorted(glob(f'{RESULTS}/resnet50_1333-0.892138.hdf5'))[-1]
# model_path = sorted(glob(f'/scratch/diogo.alves/results/baseline/resnet50*'))[-1]

print(model_path)

network = SubParts_SSD_PPN_ResNet50(class_labels = ['qr_code'],
                                     subparts_class_labels = ['fip'],
                                     input_shape = (480,480,3))


with custom_object_scope({'resnet50': tf.keras.applications.resnet50,
                        'relu6': tf.keras.layers.ReLU(6.),
                        'ppn_loss': get_ppn_loss()}):
    model = tf.keras.models.load_model(model_path)



test_qr_codes = pd.read_csv(f'test-images/src/qr_codes_offline_test.csv', dtype={'image_id': str, 'object_id': str})
test_fips = pd.read_csv(f'test-images/src/fips_offline_test.csv', dtype={'image_id': str, 'object_id': str})

batch_generator_test = SubPartsBatchGenerator(network)
batch_generator_test.add_data(dataset = test_qr_codes, subparts_dataset = test_fips, images_dir=f'test-images/src/images')

test_generator, test_size = batch_generator_test.get_generator(batch_size = 1, shuffle = False, augmentation = False, encode_output = False)
test_X, test_y_subparts, test_y_obj = [], [], []

for _ in range(test_size):
  batch_X, batch_y = next(test_generator)
  test_X.append(batch_X[0])
  test_y_subparts.append(batch_y[0][0])
  test_y_obj.append(batch_y[1][0])
  
test_X = np.array(test_X)

y_pred = model.predict(test_X, batch_size = batch_size)
y_pred = [
    subparts_output_decoder(y_pred[0], network, nms_threshold = 0.3),
    output_decoder(y_pred[1], network, nms_threshold = 0.3)
]

headers = [ 'image_id', 'obj_detected','class', 'xmin', 'ymin', 'xmax', 'ymax']

for i in range(test_size):
  img, subpart_boxes, obj_boxes = test_X[i], y_pred[0][i], y_pred[1][i]

  img = obj_boxes.draw_on_image(img, size=6)
  img = subpart_boxes.draw_on_image(img, size=4, color=[255,0,0])
  filename = f'test-images/predicted/images/{test_qr_codes.iloc[i]["image_id"]}.jpg'
  # cv.imwrite(f'test-images/predicted/image_%d.png'%i, img[..., [2,1,0]])
  cv.imwrite(filename, img[..., [2,1,0]])
