# encoding=utf-8


import time
import numpy as np
from inference import ColaClassifier

if __name__ == '__main__':
    dd = ""

    d = "/home/zhaojh/PycharmProjects/ebest_cola/exported/mobile_colav4.1_61000"
    fm = d + "/frozen_inference_graph.pb"
    mp = d + "/label_index.map"
    classifier = ColaClassifier(frozen_model_path=fm, label_index_map_path=mp)

    st = time.time()
    for i in range(1000):
        top_5 = classifier.get_top_n(image=np.zeros([64, 64, 3]), n=5)
        print(top_5)
    ed = time.time()
    print(ed - st)
