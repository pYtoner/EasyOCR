from PIL import Image
import easyocr
import onnxruntime
import torch
import numpy as np
from torch.autograd import Variable
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
import cv2


reader = easyocr.Reader(["en"], gpu=False)
# result = reader.readtext("examples/screen1.png")
# print(result)
result = reader.readtext("examples/screen2.png")
# print(result)


x = Image.open("examples/screen1.png").convert("RGB")
image_arrs = [np.array(x)]
# Image.fromarray(input[0].detach().cpu().numpy()).show()
# print(input.shape)

img_resized_list = []
# resize
for img in image_arrs:
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, 128,
                                                                  interpolation=cv2.INTER_LINEAR,
                                                                  mag_ratio=1.0)
    img_resized_list.append(img_resized)

ratio_h = ratio_w = 1 / target_ratio
# preprocessing
x = np.array([normalizeMeanVariance(n_img) for n_img in img_resized_list])
x = Variable(torch.from_numpy(x).permute(0, 3, 1, 2))  # [b,h,w,c] to [b,c,h,w]
x = x.to("cpu")


ort_session = onnxruntime.InferenceSession("detector_craft.onnx")


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
y, feature = ort_session.run(None, ort_inputs)

print(
    f"onnx outputs: y_onnx_out.shape={y.shape} feature_onnx_out.shape={feature.shape}")


estimate_num_chars = False

boxes_list, polys_list = [], []
for out in y:
    # make score and link map
    score_text = out[:, :, 0]
    score_link = out[:, :, 1]

    # Post-processing
    boxes, polys, mapper = getDetBoxes(
        score_text, score_link, 0.7, 0.4, 0.4, False, estimate_num_chars)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    if estimate_num_chars:
        boxes = list(boxes)
        polys = list(polys)
    for k in range(len(polys)):
        if estimate_num_chars:
            boxes[k] = (boxes[k], mapper[k])
        if polys[k] is None:
            polys[k] = boxes[k]
    boxes_list.append(boxes)
    polys_list.append(polys)


print(boxes_list)
# print(polys_list)
