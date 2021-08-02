from easyocr.utils import get_image_list
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2
import onnxruntime
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
import math


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


def custom_mean(x):
    print("--", x.prod())
    return x.prod() ** (2.0 / np.sqrt(len(x)))


def decode_greedy(text_index, length):
    """ convert text-index into text-label. """
    character = ['[blank]'] + list(
        "0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    ignore_idx = [0]

    texts = []
    index = 0
    for l in length:
        t = text_index[index:index + l]
        # Returns a boolean array where true is when the value is not repeated
        a = np.insert(~(t[1:] == t[:-1]), 0, True)
        # Returns a boolean array where true is when the value is not in the ignore_idx list
        b = ~np.isin(t, np.array(ignore_idx))
        # Combine the two boolean array
        c = a & b
        # Gets the corresponding character according to the saved indexes
        text = ''.join(np.array(character)[t[c.nonzero()]])
        texts.append(text)
        index += l
    return texts


def normalize_pad(img, max_size):
    img.sub_(0.5).div_(0.5)
    c, h, w = img.size()
    padded_img = torch.FloatTensor(*max_size).fill_(0)
    padded_img[:, :, :w] = img  # right pad
    if max_size[2] != w:  # add border Pad
        padded_img[:, :, w:] = img[:, :, w -
                                   1].unsqueeze(2).expand(c, h, max_size[2] - w)

    return padded_img


def process(image, imgH, imgW):
    w, h = image.size
    # TODO: adjust image's contrast maybe

    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)

    resized_image = image.resize((resized_w, imgH), Image.BICUBIC)
    resized_image = to_tensor(resized_image)

    return normalize_pad(resized_image, (1, imgH, imgW)).unsqueeze(0)


imgH = 64
img_cv_grey = cv2.imread("examples/screen2.png", cv2.IMREAD_GRAYSCALE)
print(img_cv_grey.shape)  # (26, 293)


image_list, max_width = get_image_list(
    [[6, 254, 2, 26]], [], img_cv_grey, model_height=imgH)
image = image_list[0]

# print(image[1].shape)
print(f"max_width: {max_width}")

# Image.fromarray(image[1], 'L').show()
image = process(Image.fromarray(image[1], 'L'), imgH, int(max_width))
# to_pil_image(image[0]).show()
# print(image.shape)
# print(image.mean(), image.std())

ort_session = onnxruntime.InferenceSession("recognizer.onnx")

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
preds_onnx = ort_session.run(None, ort_inputs)[0]
preds = torch.tensor(preds_onnx)
print(preds.shape)

# Select max probabilty (greedy decoding) then decode index to character
preds_size = torch.IntTensor([preds.size(1)] * 1)

######## filter ignore_char, rebalance
preds_prob = F.softmax(preds, dim=2)
preds_prob = preds_prob.cpu().detach().numpy()
pred_norm = preds_prob.sum(axis=2)
preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
preds_prob = torch.from_numpy(preds_prob).float().to("cpu")

_, preds_index = preds_prob.max(2)
preds_index = preds_index.view(-1)
# print(preds_index.data.cpu().detach().numpy().shape, preds_size.data)
preds_str = decode_greedy(
    preds_index.data.cpu().detach().numpy(), preds_size.data)
print(preds_str)

preds_prob = preds_prob.cpu().detach().numpy()
values = preds_prob.max(axis=2)
indices = preds_prob.argmax(axis=2)
preds_max_prob = []
for v, i in zip(values, indices):
    max_probs = v[i != 0]
    if len(max_probs) > 0:
        preds_max_prob.append(max_probs)
    else:
        preds_max_prob.append(np.array([0]))

for pred, pred_max_prob in zip(preds_str, preds_max_prob):
    print(len(pred_max_prob))
    confidence_score = custom_mean(pred_max_prob)
    print([pred, confidence_score])
