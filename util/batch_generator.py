# From baseline project
# @author Leonardo Blanger https://github.com/Leonardo-Blanger/subparts_ppn_keras
#
import pandas as pd
import numpy as np
import cv2 as cv
import os
import sys
from tqdm import tqdm
import pickle
import imgaug as ia
# from imgaug import BoundingBox, BoundingBoxesOnImage
from glob import glob
from keras.preprocessing.image import load_img, img_to_array

from .bounding_boxes import BoundingBox
from .metrics import IoU

class SubPartsBatchGenerator:
    def __init__(self,
                 network,
                 dataset = None,
                 subparts_dataset = None,
                 images_dir = None,
                 pickled_dataset = None,
                 channels = 'RGB'):

        self.network = network
        self.images = []
        self.objects = []
        self.subparts = []

        if not dataset is None or not pickled_dataset is None:
            self.add_data(dataset, subparts_dataset, images_dir, pickled_dataset)

    def add_data(self,
                 dataset = None,
                 subparts_dataset = None,
                 images_dir = None,
                 pickled_dataset = None,
                 channels = 'RGB'):

        if not pickled_dataset is None and os.path.exists(pickled_dataset):
            with open(pickled_dataset, 'rb') as f:
                images, objects, subparts = pickle.load(f)

            if images.shape[1:] != self.network.input_shape:
                raise Exception('The shape of the images in %s is '+
                                'not compatible with the network')
        else:
            if dataset is None:
                raise Exception('At least one of dataset or pickled_dataset must be provided')

            if isinstance(dataset, str):
                dataset = pd.read_csv(dataset, dtype={
                                                'image_id': str,
                                                'object_id': str})

            if isinstance(subparts_dataset, str):
                subparts_dataset = pd.read_csv(subparts_dataset, dtype={
                                                                    'image_id': str,
                                                                    'object_id': str})

            input_height, input_width = self.network.input_shape[:2]
            images = {}
            objects = {}
            subparts = {}

            for i in tqdm(range(dataset.shape[0]), desc='Preprocessing Dataset'):
                entry = dataset.loc[i]
                img_id = str(entry['image_id'])
                obj_id = str(entry['object_id'])

                filepath = glob(os.path.join(images_dir, img_id + '*'))[0]
                image_height = entry['image_height']
                image_width = entry['image_width']

                if not img_id in images:
                    #img = img_to_array(load_img(filepath, target_size=(input_height, input_width)))
                    img = cv.resize(cv.imread(filepath), (input_width, input_height))
                    images[img_id] = img
                    objects[img_id] = []
                    subparts[img_id] = []
                    del img

                obj_class = self.network.class_labels.index(entry['class'])
                xmin = entry['xmin'] * float(input_width) / image_width
                ymin = entry['ymin'] * float(input_height) / image_height
                xmax = entry['xmax'] * float(input_width) / image_width
                ymax = entry['ymax'] * float(input_height) / image_height
                objects[img_id].append([obj_class, xmin, ymin, xmax, ymax])

                obj_subparts = subparts_dataset.loc[subparts_dataset['object_id'] == obj_id].reset_index()

                for j in range(len(obj_subparts)):
                    subpart = obj_subparts.loc[j]

                    subpart_class = self.network.subparts_class_labels.index(subpart['class'])
                    xmin = subpart['xmin'] * float(input_width) / image_width
                    ymin = subpart['ymin'] * float(input_height) / image_height
                    xmax = subpart['xmax'] * float(input_width) / image_width
                    ymax = subpart['ymax'] * float(input_height) / image_height
                    subparts[img_id].append([subpart_class, xmin, ymin, xmax, ymax])

            images = np.array(list(images.values()))

            for img_id in objects:
                objects[img_id] = np.array(objects[img_id])
            objects = list(objects.values())

            for img_id in subparts:
                subparts[img_id] = np.array(subparts[img_id])
            subparts = list(subparts.values())

            channels = channels.lower()
            if channels == 'bgr':
                pass
            elif channels == 'rgb':
                images = images[..., [2,1,0]]
            else:
                raise Exception('Channel format not supported: %s' % channels)

            if not pickled_dataset is None:
                with open(pickled_dataset, 'wb') as f:
                    pickle.dump((images, objects, subparts), f)

        if len(self.images) == 0:
            self.images = images
        else:
            self.images = np.concatenate([self.images, images], axis = 0)
        self.objects += objects
        self.subparts += subparts

    def get_generator(self, batch_size = 32,
                      shuffle = False,
                      encode_output = False,
                      augmentation = None):

        def generator(images, objects, subparts):
            batch_start = 0

            if shuffle:
                perm = np.random.permutation(len(images))
                images = images[perm]
                objects = [objects[i] for i in perm]
                subparts = [subparts[i] for i in perm]

            while True:
                if batch_start + batch_size > len(images):
                    if shuffle:
                        perm = np.random.permutation(len(images))
                        images = images[perm]
                        objects = [objects[i] for i in perm]
                        subparts = [subparts[i] for i in perm]
                    batch_start = 0

                batch_X = images[batch_start : batch_start+batch_size]
                batch_y_objects = [
                    ia.BoundingBoxesOnImage([
                        BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=self.network.class_labels[int(label)])
                        for (label, x1, y1, x2, y2) in img_boxes
                    ], shape = self.network.input_shape)
                    for img_boxes in objects[batch_start : batch_start+batch_size]
                ]
                batch_y_subparts = [
                    ia.BoundingBoxesOnImage([
                        BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=self.network.subparts_class_labels[int(label)])
                        for (label, x1, y1, x2, y2) in img_boxes
                    ], shape = self.network.input_shape)
                    for img_boxes in subparts[batch_start : batch_start+batch_size]
                ]
                batch_start += batch_size

                if augmentation:
                    batch_X, batch_y_objects, batch_y_subparts = self.augment(
                        batch_X, batch_y_objects, batch_y_subparts, augmentation)

                if encode_output:
                    batch_y_objects = output_encoder(batch_y_objects, self.network)
                    batch_y_subparts = subparts_output_encoder(batch_y_subparts, self.network)

                batch_y = [batch_y_subparts, batch_y_objects]

                yield batch_X, batch_y

        return generator(self.images, self.objects, self.subparts), len(self.images)


    def augment(self, images, object_boxes, subpart_boxes, augmentation_seq, max_tries = 1):
        for _ in range(max_tries):
            try:
                seq_det = augmentation_seq.to_deterministic()
                _object_boxes = seq_det.augment_bounding_boxes(object_boxes)
                _subpart_boxes = seq_det.augment_bounding_boxes(subpart_boxes)
                _images = seq_det.augment_images(np.copy(images))
                object_boxes = _object_boxes
                subpart_boxes = _subpart_boxes
                images = _images
                break
            except:
                continue

        object_boxes = [img_boxes.remove_out_of_image().cut_out_of_image() for img_boxes in object_boxes]
        subpart_boxes = [img_boxes.remove_out_of_image().cut_out_of_image() for img_boxes in subpart_boxes]

        return images, object_boxes, subpart_boxes

def output_encoder(ground_truth, network, neg_iou_threshold = 0.3, pos_iou_threshold = 0.5):
    anchor_xmin = network.anchor_xmin
    anchor_ymin = network.anchor_ymin
    anchor_xmax = network.anchor_xmax
    anchor_ymax = network.anchor_ymax
    num_anchors = anchor_xmin.shape[0]
    batch_output = []

    # For each item in the batch
    for boxes in ground_truth:
        num_gt = len(boxes.bounding_boxes)
        output = np.zeros((num_anchors, network.num_classes + 4))

        if num_gt == 0:
            output[:, network.background_id] = 1.0
            batch_output.append(output)
            continue

        ious = []

        for box in boxes.bounding_boxes:
            ious.append(IoU(box.x1, box.y1, box.x2, box.y2,
                            anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))
        ious = np.array(ious)

        matches = [[] for _ in range(num_gt)]

        def find_best_match_for_gts(ious):
            ious = ious.copy()

            for _ in range(num_gt):
                best_gt = np.argmax(np.max(ious, axis = 1))
                best_anchor = np.argmax(ious[best_gt, :])

                matches[best_gt].append(best_anchor)
                ious[best_gt, :] = -1.0
                ious[:, best_anchor] = -1.0

        def find_best_match_for_anchors(ious):
            for anchor_index in range(num_anchors):
                if ious[0, anchor_index] < 0.0: continue
                best_gt = np.argmax(ious[:, anchor_index])

                if ious[best_gt, anchor_index] >= pos_iou_threshold:
                    matches[best_gt].append(anchor_index)
                elif ious[best_gt, anchor_index] < neg_iou_threshold:
                    output[anchor_index, network.background_id] = 1.0

        find_best_match_for_gts(ious.copy())
        ious[:, [match[0] for match in matches]] = -1.0
        find_best_match_for_anchors(ious)

        for i, box in enumerate(boxes.bounding_boxes):
            box_cx = (box.x1 + box.x2) * 0.5
            box_cy = (box.y1 + box.y2) * 0.5
            box_w = box.x2 - box.x1
            box_h = box.y2 - box.y1

            if box_w < 1.0 or box_h < 1.0: continue

            anchor_cx = network.anchor_cx[matches[i]]
            anchor_cy = network.anchor_cy[matches[i]]
            anchor_w  = network.anchor_width[matches[i]]
            anchor_h  = network.anchor_height[matches[i]]

            output[matches[i], network.class_labels.index(box.label)] = 1.0
            output[matches[i], -4] = (box_cx - anchor_cx) / anchor_w
            output[matches[i], -3] = (box_cy - anchor_cy) / anchor_h
            output[matches[i], -2] = np.log(box_w / anchor_w)
            output[matches[i], -1] = np.log(box_h / anchor_h)

        batch_output.append(output)

    return np.array(batch_output)

def output_decoder(batch_output, network, conf_threshold = 0.5, nms_threshold = 0.5):
    predicted_boxes = []

    for output in batch_output:
        predictions = np.where(np.logical_and(
            np.argmax(output[:,:-4], axis=1) != network.background_id, np.max(output[:,:-4], axis=1) >= conf_threshold
        ))[0]

        class_id = np.argmax(output[predictions, :-4], axis=1)
        conf = np.max(output[predictions, :-4], axis=1)

        anchor_cx = network.anchor_cx[predictions]
        anchor_cy = network.anchor_cy[predictions]
        anchor_w  = network.anchor_width[predictions]
        anchor_h  = network.anchor_height[predictions]

        box_cx = output[predictions, -4] * anchor_w + anchor_cx
        box_cy = output[predictions, -3] * anchor_h + anchor_cy
        box_w  = np.exp(output[predictions, -2]) * anchor_w
        box_h  = np.exp(output[predictions, -1]) * anchor_h

        xmin = box_cx - box_w * 0.5
        ymin = box_cy - box_h * 0.5
        xmax = box_cx + box_w * 0.5
        ymax = box_cy + box_h * 0.5

        class_id = np.expand_dims(class_id, axis = -1)
        conf = np.expand_dims(conf, axis = -1)
        xmin = np.expand_dims(xmin, axis = -1)
        ymin = np.expand_dims(ymin, axis = -1)
        xmax = np.expand_dims(xmax, axis = -1)
        ymax = np.expand_dims(ymax, axis = -1)

        boxes = np.concatenate([class_id, conf, xmin, ymin, xmax, ymax], axis = -1)

        if boxes.shape[0] == 0:
            predicted_boxes.append(ia.BoundingBoxesOnImage([], shape = network.input_shape[:2]))
            continue

        # NMS
        nms_boxes = []

        for class_id in range(network.num_classes):
            if class_id == network.background_id: continue

            class_predictions = np.array([box for box in boxes if box[0] == class_id])
            if class_predictions.shape[0] == 0: continue

            class_predictions = class_predictions[np.flip(np.argsort(class_predictions[:,1], axis=0), axis=0)]
            nms_class_boxes = np.array([class_predictions[0]])

            for box in class_predictions[1:]:
                xmin1, ymin1, xmax1, ymax1 = box[2:]
                xmin2, ymin2, xmax2, ymax2 = [nms_class_boxes[:,i] for i in range(2,6)]

                if np.all(IoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) < nms_threshold):
                    nms_class_boxes = np.concatenate([nms_class_boxes, [box]], axis=0)

            [nms_boxes.append(BoundingBox(
                x1=box[2], y1=box[3], x2=box[4], y2=box[5], label=network.class_labels[int(box[0])], confidence=box[1]
            )) for box in nms_class_boxes]

        predicted_boxes.append(ia.BoundingBoxesOnImage(nms_boxes, shape = network.input_shape[:2]))

    return predicted_boxes

def subparts_output_encoder(ground_truth, network, neg_iou_threshold = 0.3, pos_iou_threshold = 0.5):
    anchor_xmin = network.subparts_anchor_xmin
    anchor_ymin = network.subparts_anchor_ymin
    anchor_xmax = network.subparts_anchor_xmax
    anchor_ymax = network.subparts_anchor_ymax
    num_anchors = anchor_xmin.shape[0]
    batch_output = []

    # For each item in the batch
    for boxes in ground_truth:
        num_gt = len(boxes.bounding_boxes)
        output = np.zeros((num_anchors, network.num_subpart_classes + 4))

        if num_gt == 0:
            output[:, network.subparts_background_id] = 1.0
            batch_output.append(output)
            continue

        ious = []

        for box in boxes.bounding_boxes:
            ious.append(IoU(box.x1, box.y1, box.x2, box.y2,
                            anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))
        ious = np.array(ious)

        matches = [[] for _ in range(num_gt)]

        def find_best_match_for_gts(ious):
            ious = ious.copy()

            for _ in range(num_gt):
                best_gt = np.argmax(np.max(ious, axis = 1))
                best_anchor = np.argmax(ious[best_gt, :])

                matches[best_gt].append(best_anchor)
                ious[best_gt, :] = -1.0
                ious[:, best_anchor] = -1.0

        def find_best_match_for_anchors(ious):
            for anchor_index in range(num_anchors):
                if ious[0, anchor_index] < 0.0: continue
                best_gt = np.argmax(ious[:, anchor_index])

                if ious[best_gt, anchor_index] >= pos_iou_threshold:
                    matches[best_gt].append(anchor_index)
                elif ious[best_gt, anchor_index] < neg_iou_threshold:
                    output[anchor_index, network.subparts_background_id] = 1.0

        find_best_match_for_gts(ious.copy())
        ious[:, [match[0] for match in matches]] = -1.0
        find_best_match_for_anchors(ious)

        for i, box in enumerate(boxes.bounding_boxes):
            if box.area < 1.0: continue

            box_cx = (box.x1 + box.x2) * 0.5
            box_cy = (box.y1 + box.y2) * 0.5
            box_w = box.x2 - box.x1
            box_h = box.y2 - box.y1

            anchor_cx = network.subparts_anchor_cx[matches[i]]
            anchor_cy = network.subparts_anchor_cy[matches[i]]
            anchor_w  = network.subparts_anchor_width[matches[i]]
            anchor_h  = network.subparts_anchor_height[matches[i]]

            output[matches[i], network.subparts_class_labels.index(box.label)] = 1.0
            output[matches[i], -4] = (box_cx - anchor_cx) / anchor_w
            output[matches[i], -3] = (box_cy - anchor_cy) / anchor_h
            output[matches[i], -2] = np.log(box_w / anchor_w)
            output[matches[i], -1] = np.log(box_h / anchor_h)

        batch_output.append(output)

    return np.array(batch_output)

def subparts_output_decoder(batch_output, network, conf_threshold = 0.5, nms_threshold = 0.5):
    predicted_boxes = []

    for output in batch_output:
        predictions = np.where(np.logical_and(
            np.argmax(output[:,:-4], axis=1) != network.subparts_background_id, np.max(output[:,:-4], axis=1) >= conf_threshold
        ))[0]

        class_id = np.argmax(output[predictions, :-4], axis=1)
        conf = np.max(output[predictions, :-4], axis=1)

        anchor_cx = network.subparts_anchor_cx[predictions]
        anchor_cy = network.subparts_anchor_cy[predictions]
        anchor_w  = network.subparts_anchor_width[predictions]
        anchor_h  = network.subparts_anchor_height[predictions]

        box_cx = output[predictions, -4] * anchor_w + anchor_cx
        box_cy = output[predictions, -3] * anchor_h + anchor_cy
        box_w  = np.exp(output[predictions, -2]) * anchor_w
        box_h  = np.exp(output[predictions, -1]) * anchor_h

        xmin = box_cx - box_w * 0.5
        ymin = box_cy - box_h * 0.5
        xmax = box_cx + box_w * 0.5
        ymax = box_cy + box_h * 0.5

        class_id = np.expand_dims(class_id, axis = -1)
        conf = np.expand_dims(conf, axis = -1)
        xmin = np.expand_dims(xmin, axis = -1)
        ymin = np.expand_dims(ymin, axis = -1)
        xmax = np.expand_dims(xmax, axis = -1)
        ymax = np.expand_dims(ymax, axis = -1)

        boxes = np.concatenate([class_id, conf, xmin, ymin, xmax, ymax], axis = -1)

        if boxes.shape[0] == 0:
            predicted_boxes.append(ia.BoundingBoxesOnImage([], shape = network.input_shape[:2]))
            continue

        # NMS
        nms_boxes = []

        for class_id in range(network.num_subpart_classes):
            if class_id == network.subparts_background_id: continue

            class_predictions = np.array([box for box in boxes if box[0] == class_id])
            if class_predictions.shape[0] == 0: continue

            class_predictions = class_predictions[np.flip(np.argsort(class_predictions[:,1], axis=0), axis=0)]
            nms_class_boxes = np.array([class_predictions[0]])

            for box in class_predictions[1:]:
                xmin1, ymin1, xmax1, ymax1 = box[2:]
                xmin2, ymin2, xmax2, ymax2 = [nms_class_boxes[:,i] for i in range(2,6)]

                if np.all(IoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) < nms_threshold):
                    nms_class_boxes = np.concatenate([nms_class_boxes, [box]], axis=0)

            [nms_boxes.append(BoundingBox(
                x1=box[2], y1=box[3], x2=box[4], y2=box[5], label=network.subparts_class_labels[int(box[0])], confidence=box[1]
            )) for box in nms_class_boxes]

        predicted_boxes.append(ia.BoundingBoxesOnImage(nms_boxes, shape = network.input_shape[:2]))

    return predicted_boxes