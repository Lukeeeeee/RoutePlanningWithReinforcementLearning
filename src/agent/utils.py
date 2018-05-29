import itertools
from PIL import Image
import json


def slice_queue(q, left, right):
    return list(itertools.islice(q, left, right))


def show_pic(img):
    im = Image.fromarray(img)
    im.show()


def load_json(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
        return res
