import json
import time
from typing import List, Tuple

import jsonpickle

import numpy as np
import tensorflow as tf

from Games.financial_model.archive.Archive import Archive
from Games.financial_model.archive.structures.Timeline import TimelinePointer, Timeline


class Normalizer:
    def __init__(self, path: str):
        self.path = path
        self.dict = dict()
        self.load()

    def build_internal(self, sub_dict: dict, target: dict):
        for key in sub_dict:
            e = sub_dict[key]
            if isinstance(e, Timeline):
                for o in e.list:
                    if not (isinstance(o.description, float) or isinstance(o.description, int)):
                        break
                    if key not in target:
                        target[key] = []
                    target[key].append(o.description)
            else:
                if key not in target:
                    target[key] = dict()
                self.build_internal(e, target[key])

    def aggregate_internal(self, sub_dict: dict):
        for key in sub_dict:
            e = sub_dict[key]
            if isinstance(e, List):
                e_mean = np.mean(e, dtype=np.float32)
                e_std = np.std(e, dtype=np.float32)
                sub_dict[key] = (
                    e_mean,
                    e_std,
                )
            else:
                self.aggregate_internal(e)

    def build(self, archive: Archive):
        self.dict = dict()
        for symbol in archive.dict.keys():
            self.build_internal(archive.dict[symbol], self.dict)

        self.aggregate_internal(self.dict)

    def apply_internal(self, sub_dict, target):
        for key in sub_dict:
            if key not in target:
                continue
            e = sub_dict[key]
            if isinstance(e, Tuple):
                mean, std = e
                list = target[key].list
                for i in range(len(list)):
                    if std == 0.0:
                        list[i].description = 0.0
                    else:
                        list[i].description = (list[i].description - mean) / std
            else:
                self.apply_internal(e, target[key])

    def apply(self, archive: Archive):
        for symbol in archive.dict.keys():
            self.apply_internal(self.dict, archive.dict[symbol])

    def load(self):
        start = time.time()
        try:
            f = open(self.path)
            data = json.load(f)
            self.dict = jsonpickle.decode(data)
        except IOError:
            tf.print("Normalizer file error.")
        end = time.time()
        tf.print(f'Normalizer load took: {end - start}')

    def save(self):
        start = time.time()
        data = jsonpickle.encode(self.dict)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        end = time.time()
        tf.print(f'Normalizer save took: {end - start}')
