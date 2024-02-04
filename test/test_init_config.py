import unittest
from plain.train import load_config_dict, init_config_object


d = {
    "a": 1,
    "b": {"c": "d", "e": 2, "f": None},
    "g": [],
    "h": [1, "i"],
    "j": [1, "k", {}],
    "l": [
        1,
        "m",
        {
            "n": [3],
            "o": "p",
            "q": {
                "r": "s",
                "t": [
                    "u",
                    5,
                    {"v": "w"},
                ],
                "x": ("z", 1),
            },
        },
    ],
}


class Testconfig(unittest.TestCase):
    def test_config_initiation(self):
        config = init_config_object(d)
        self.assertEqual(config.a, 1)
        self.assertEqual(config.l[2].q.t[0], "u")


if __name__ == "__main__":
    unittest.main()
