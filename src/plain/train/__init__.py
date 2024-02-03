from plain.train.trainer import Trainer, load_config_dict, init_config_object


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
