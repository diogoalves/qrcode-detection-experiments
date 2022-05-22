import tensorflow as tf
import numpy as np
import imgaug as ia
import pandas as pd

from tensorflow.keras.callbacks import Callback
from dvclive import Live
from dvclive.keras import DvcLiveCallback

from imgaug import augmenters as iaa

from util.metrics import AP, get_ppn_loss
from util.network import SubParts_SSD_PPN_ResNet50
from util.batch_generator import SubPartsBatchGenerator, output_decoder, subparts_output_decoder

live = Live('training_metrics')

np.random.seed(1)

DATASET = './data'
RESULTS = '/scratch/diogo.alves/results/baseline'
batch_size = 8
epochs = 100

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('\nSetup complete!')


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
augmentation_seq = iaa.Sequential(
    [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images

        # crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),

        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's chanell with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)


class PrintMetricsOnEnd(Callback):
    def __init__(self):
        super(PrintMetricsOnEnd, self).__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params:
            if k in logs:
                print('%s: %.10f' % (k, logs[k]))
        print()

class EvaluateValidMeanAP(Callback):
    def __init__(self):
        super(EvaluateValidMeanAP, self).__init__()
        self.best_mAP = 0
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred = network.model.predict(valid_X, batch_size = batch_size, verbose = 0)
        y_pred = [
            subparts_output_decoder(y_pred[0], network, nms_threshold = 0.3),
            output_decoder(y_pred[1], network, nms_threshold = 0.3)
        ]
        
        APs = [
            AP(valid_y_subparts, y_pred[0]),
            AP(valid_y_obj, y_pred[1])
        ]
        

        print('Validation APs:', APs)
        print()

        live.log('val_main_ap', APs[1].astype(float) )
        live.log('val_subparts_ap', APs[0].astype(float) )
        live.next_step()

        
        if APs[1] > self.best_mAP:
            self.best_mAP = APs[1]
            network.model.save(f'{RESULTS}/resnet50_{epoch:03d}-{APs[1]:.6f}.hdf5')
            print('Saving... ')



network = SubParts_SSD_PPN_ResNet50(class_labels = ['qr_code'],
                                    subparts_class_labels = ['fip'],
                                    input_shape = (480,480,3))

network.model.summary()

#### TRAIN DATA
train_qr_codes = pd.read_csv(f'{DATASET}/qr_codes_train.csv', dtype={'image_id': str, 'object_id': str})
valid_qr_codes = pd.read_csv(f'{DATASET}/qr_codes_valid.csv', dtype={'image_id': str, 'object_id': str})

train_fips = pd.read_csv(f'{DATASET}/fips_train.csv', dtype={'image_id': str, 'object_id': str})
valid_fips = pd.read_csv(f'{DATASET}/fips_valid.csv', dtype={'image_id': str, 'object_id': str})

batch_generator_train = SubPartsBatchGenerator(network)
batch_generator_valid = SubPartsBatchGenerator(network)

batch_generator_train.add_data(dataset = train_qr_codes, subparts_dataset = train_fips, images_dir=f'{DATASET}/images')
batch_generator_valid.add_data(dataset = valid_qr_codes, subparts_dataset = valid_fips, images_dir=f'{DATASET}/images')

valid_generator, valid_size = batch_generator_valid.get_generator(batch_size = 1, shuffle = False, augmentation = False, encode_output = False)
valid_X, valid_y_subparts, valid_y_obj = [], [], []

for _ in range(valid_size):
    batch_X, batch_y = next(valid_generator)
    valid_X.append(batch_X[0])
    valid_y_subparts.append(batch_y[0][0])
    valid_y_obj.append(batch_y[1][0])

valid_X = np.array(valid_X)

train_generator, train_size = batch_generator_train.get_generator(batch_size=batch_size, shuffle=True, augmentation=augmentation_seq, encode_output=True)
valid_generator, valid_size = batch_generator_valid.get_generator(batch_size = batch_size, shuffle = False, augmentation = False, encode_output = True)

print('%d training images' % train_size)
print('%d validation images' % valid_size)

batch_X, batch_y = next(train_generator)
print(batch_X.shape)
print(batch_y[0].shape, batch_y[1].shape)

network.model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-4, decay=1e-4),
    loss = {'subparts_output': get_ppn_loss(), 'main_output': get_ppn_loss()},
    loss_weights = {'subparts_output': 1.0, 'main_output': 1.0}
)

network.model.fit(train_generator,
                        # steps_per_epoch = train_size // batch_size,
                        steps_per_epoch = 50,
                        epochs = epochs,
                        validation_data = valid_generator,
                        validation_steps = valid_size // batch_size,
                        verbose = 1,
                        max_queue_size = 200,
                        callbacks = [
                            PrintMetricsOnEnd(),
                            DvcLiveCallback(path='training_metrics'),
                            EvaluateValidMeanAP()
                        ])  

