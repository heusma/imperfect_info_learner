import json
import time

import tensorflow as tf

import jsonpickle


class Archive:
    def __init__(self, path: str):
        self.path = path
        self.dict = dict()
        self.load()

    def load(self):
        start = time.time()
        try:
            f = open(self.path)
            data = json.load(f)
            self.dict = jsonpickle.decode(data)
        except IOError:
            tf.print("Archive file error.")
        end = time.time()
        tf.print(f'archive load took: {end - start}')

    def save(self):
        start = time.time()
        data = jsonpickle.encode(self.dict)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        end = time.time()
        tf.print(f'archive save took: {end - start}')

    def __getitem__(self, path):
        node = self.dict
        for key in path:
            node = node[key]
        return node

    def __setitem__(self, path, value):
        node = self.dict
        for key in path[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = dict()
            node = node[key]
        node[path[-1]] = value
        return node
