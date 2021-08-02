from easyocr.recognition import get_recognizer, AlignCollate
from easyocr.utils import get_image_list
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch import nn
import cv2
import onnx
import onnxruntime


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))


network_params = {
    'input_channel': 1,
    'output_channel': 256,
    'hidden_size': 256
}

recognizer, converter = get_recognizer(
    "generation2",
    network_params,
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    {},
    {'en': 'easyocr/dict/en.txt'},
    "/home/.EasyOCR//model/english_g2.pth",
)

imgH = 64
img_cv_grey = cv2.imread("examples/screen2.png", cv2.IMREAD_GRAYSCALE)
print(img_cv_grey.shape)  # (26, 293)


image_list, max_width = get_image_list(
    [[6, 254, 2, 26]], [], img_cv_grey, model_height=imgH)
image = image_list[0]

print(f"max_width: {max_width}")

process = AlignCollate(imgH=imgH, imgW=int(
    max_width), keep_ratio_with_pad=True)
image = process([Image.fromarray(image[1], 'L')])

recognizer.eval()
with torch.no_grad():
    print(image.shape)  # torch.Size([1, 1, 64, 704])
    preds = recognizer(image)
    print(preds.shape)  # torch.Size([1, 175, 97])

    torch.onnx.export(
        model=recognizer,
        args=(image,),
        f="model.onnx",
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: "height", 3: "width"},
            'output': {1: "dim1", 2: "dim2"}
        })

    # verify exported onnx model
    recognizer_onnx = onnx.load("model.onnx")
    onnx.checker.check_model(recognizer_onnx)
    # print(f"Model Inputs:\n {recognizer_onnx.graph.input}\n{'*'*80}")
    # print(f"Model Outputs:\n {recognizer_onnx.graph.output}\n{'*'*80}")

    ort_session = onnxruntime.InferenceSession("model.onnx")

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    preds_onnx = ort_session.run(None, ort_inputs)[0]
    preds = torch.tensor(preds_onnx)

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
    preds_str = converter.decode_greedy(
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
        confidence_score = custom_mean(pred_max_prob)
        print([pred, confidence_score])
