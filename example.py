from PIL import Image
import easyocr
import onnxruntime
import torch
import numpy as np
from torch.autograd import Variable
from easyocr.imgproc import resize_aspect_ratio
import cv2
import math
import warnings
warnings.filterwarnings('ignore')


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255

        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)

    return det


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


def normalize(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0,
                     mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0,
                     variance[2] * 255.0], dtype=np.float32)
    return img


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


reader = easyocr.Reader(["en"], gpu=False)
# result = reader.readtext("examples/screen1.png")
# print(result)
result = reader.readtext("examples/japanese.jpg")
# print(result)


x = Image.open("examples/japanese.jpg").convert("RGB")
image = np.array(x)
# Image.fromarray(image_arrs[0]).show()

img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
    image, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0)

ratio_h = ratio_w = 1 / target_ratio

x = normalize(img_resized)[None, ...]
x = Variable(torch.from_numpy(x).permute(0, 3, 1, 2)).to(
    "cpu")  # [b,h,w,c] to [b,c,h,w]


ort_session = onnxruntime.InferenceSession("detector_craft.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
y, feature = ort_session.run(None, ort_inputs)

boxes_list = []
for out in y:
    # make score and link map
    score_text = out[:, :, 0]
    score_link = out[:, :, 1]

    # Post-processing
    boxes = getDetBoxes_core(score_text, score_link, 0.7, 0.4, 0.4, False)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    boxes_list.append(boxes)


result = []
for polys in boxes_list:
    single_img_result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        single_img_result.append(poly)
    result.append(single_img_result)

print(result)
