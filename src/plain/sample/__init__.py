from plain.train import init_config_object
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
import tomli
import json
from collections import namedtuple
from pathlib import Path
import sys
from plain.train import load_config_dict


class Sampler:
    def __init__(self, config_dict, data_class, model_class):
        self.config_dict = config_dict
        self.config = init_config_object(self.config_dict)
        self.device = self.config.device
        self.load_checkpoint()
        self.init_state()
        self.data = data_class(self.config)
        self.model_class = model_class
        self.init_model()

    @staticmethod
    def iterable_to_device(iterable, device):
        return tuple(x.to(device) for x in iterable)

    def load_checkpoint(self):
        checkpoint_path = Path("checkpoint") / (
            self.config.experiment_name + ".ckpt"
        )
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

    def init_state(self):
        self.state = SimpleNamespace()
        if self.config.init_from == "checkpoint":
            self.state = self.checkpoint["state"]
            return

    def init_model(self):
        model = self.model_class(self.config)
        if self.config.init_from == "pretrained":
            model.copy_pretrained()
        elif self.config.init_from == "checkpoint":
            model.load_checkpoint(self.checkpoint["model"])
        self.model = model.to(self.device)
        if self.config.freeze:
            self.model.freeze()

    @torch.no_grad()
    def sample(self):
        self.model.eval()
        while True:
            ids = self.data.get_data_for_sampling()
            ids = ids.to(self.device)
            prediction = self.model.sample(ids)
            print("input")
            print(self.data.decode(ids))
            print("real output")
            print(self.data.decode(prediction))
        self.model.train()


def sample(data_class, model_class):
    config_file_path = sys.argv[2]
    config_dict = load_config_dict(config_file_path)
    t = Sampler(config_dict, data_class, model_class)
    t.sample()
