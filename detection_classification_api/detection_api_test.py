# encoding=utf-8

import tqdm
from PIL import Image

from detection_api import ColaDetector

detector = ColaDetector("10.18.103.211", 9000, "ebest_carlsberg", 1512, 512)
image = Image.open(
    "/home/zhaojh/Temporary/Cola_EBest/3.8_demo_test_images/0ae4a74a0aff11e89cdb00163e0028a9.jpg")

t0 = detector.detection_only(image)
print(t0)

# import scipy.io as sio
#
# result0 = sio.loadmat("result0.mat")
# result1 = sio.loadmat("result1.mat")
# result0 = {k: v for k, v in result0.items() if not k.startswith("__")}
# result1 = {k: v for k, v in result1.items() if not k.startswith("__")}
# result0["scores"] = result0["scores"].flatten()
# result1["scores"] = result1["scores"].flatten()
#
# overlap = (500 / 1000, (1000 - 500) / (1440 - 500))
# det_result0, det_result1, num = detector.merge_overlap(result0, result1, overlap)
# print(det_result0, det_result1, num)

# for i in tqdm.tqdm(range(10)):
#     t0 = detector.detection_classify(image)
#
# for i in tqdm.tqdm(range(10)):
#     t1 = detector.classify_only(image)
#
# for i in tqdm.tqdm(range(10)):
#     t2 = detector.detection_only(image)
