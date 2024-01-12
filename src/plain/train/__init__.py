import json
from collections import namedtuple
import tomllib
from plain.train.trainer import Trainer


# https://stackoverflow.com/a/34997118/17749529
# https://stackoverflow.com/a/15882327/17749529
# namedtuple for immutability
def create_config(path):
    with open(path, "rb") as f:
        dictionary = tomllib.load(f)
    config = json.loads(
        json.dumps(dictionary),
        object_hook=lambda d: namedtuple("Config", d.keys())(*d.values()),
    )
    return dictionary, config


# d = {
#     "a": 1,
#     "b": {"c": "d", "e": 2, "f": None},
#     "g": [],
#     "h": [1, "i"],
#     "j": [1, "k", {}],
#     "l": [
#         1,
#         "m",
#         {
#             "n": [3],
#             "o": "p",
#             "q": {
#                 "r": "s",
#                 "t": [
#                     "u",
#                     5,
#                     {"v": "w"},
#                 ],
#                 "x": ("z", 1),
#             },
#         },
#     ],
# }
# c = config(d)
# print(c.a, c.l[2].q.t)
# print(c)
