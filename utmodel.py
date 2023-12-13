
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
import time
from PIL import Image

import detectron2

import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects import point_rend


def ut_load_pr_model(saved_model_weights, threshold=0.05):
    cfg = get_cfg()
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("/home/chenwei/src/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = saved_model_weights
    model = DefaultPredictor(cfg)
    return model


def im_get_fb_prediction(model, im):
    """ im is a numpy RGB image,
        can be uint8, or float in range [0, 1] or [0, 255] """
    t0 = time.time()
    im2 = im[:, :, ::-1].copy()  # to BGR
    pred = model(im2)
    t = time.time() - t0
    print('  model prediction in %.4f seconds' % t)
    pred_scores = list(pred['instances'].scores.detach().cpu().numpy())
    if not pred_scores:
        return ([], [])

    masks = pred['instances'].pred_masks.detach().cpu().numpy()
    return (masks, [float(x) for x in pred_scores])


def get_fb_prediction(model, img):
    """ img is a PIL RGB image,
        can be uint8, or float in range [0, 1] or [0, 255] """
    img = img.convert('RGB')
    im = np.asarray(img)
    return im_get_fb_prediction(model, im)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
            min_size=800,max_size=1333,box_detections_per_img=400)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def ut_load_model_to_gpu(saved_gpu_model):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using %s' % device)
    num_classes = 2
    gpu_model = get_model_instance_segmentation(num_classes)
    gpu_model.load_state_dict(torch.load(saved_gpu_model))
    gpu_model.to(device)
    return gpu_model


def ut_load_checkpoint_model_to_gpu(saved_checkpoint_model):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using %s' % device)
    num_classes = 2
    gpu_model = get_model_instance_segmentation(num_classes)

    checkpoint = torch.load(saved_checkpoint_model, map_location=device)
    gpu_model.load_state_dict(checkpoint['model'])
    print('checkpoint model loaded')
    gpu_model.to(device)
    return gpu_model


def ut_load_checkpoint_model_to_cpu(saved_checkpoint_model):
    device = torch.device('cpu')
    print('Using %s' % device)
    num_classes = 2
    gpu_model = get_model_instance_segmentation(num_classes)

    checkpoint = torch.load(saved_checkpoint_model, map_location=device)
    gpu_model.load_state_dict(checkpoint['model'])
    print('checkpoint model loaded')
    gpu_model.to(device)
    return gpu_model


def ut_load_model_to_cpu(saved_model):
    device = torch.device('cpu')
    print('Using %s' % device)
    num_classes = 2

    cpu_model = get_model_instance_segmentation(num_classes)
    cpu_model.load_state_dict(torch.load(saved_model, map_location=device))
    return cpu_model


def get_prediction(model, img, threshold):
    t0 = time.time()
    target = {}
    transform = T.Compose([T.ToTensor()])
    img, _ = transform(img, target)

    if torch.cuda.is_available():
        img = img.to('cuda')
    pred = model([img])
    t1 = time.time() - t0
    print('    model prediction in %f seconds' % t1)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    #print(pred_score)
    if not pred_score:
        #return [], [], []
        return []

    pred_score_good = []
    for i in range(len(pred_score)):
        if pred_score[i] > threshold:
            pred_score_good.append(i)

    if not pred_score_good:
        #return [], [], []
        return []

    pred_t = pred_score_good[-1]
    masks = (pred[0]['masks']>0.5).detach().cpu().numpy()
    n,_,h,w = masks.shape
    masks = masks.reshape((n,h,w))

    pred_class = []
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    print('    model prediction found %d candidate masks' % len(masks))
    return masks

