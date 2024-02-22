import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
import tomli
import json
from collections import namedtuple
from pathlib import Path
import sys

RJUST_WIDTH = 20


class StateLog:
    def __init__(self):
        self.previous_keys = None

    def log(self, keys, vals):
        vals = "".join([f"{v}".rjust(RJUST_WIDTH) for v in vals])
        if keys == self.previous_keys:
            print(vals)
            return
        self.previous_keys = keys
        keys = "".join([k.rjust(RJUST_WIDTH) for k in keys])
        print(f"\n{keys}\n{vals}")


state_log = StateLog()


def pretty_tokens(batch):
    print([[x.rjust(15) for x in sequence] for sequence in batch])


def load_config_dict(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def init_config_object(config_dict):
    # https://stackoverflow.com/a/34997118/17749529
    # https://stackoverflow.com/a/15882327/17749529
    # namedtuple for immutability
    return json.loads(
        json.dumps(config_dict),
        object_hook=lambda d: namedtuple("Config", d.keys())(*d.values()),
    )


class Trainer:
    def __init__(self, config_dict, data_class, model_class):
        self.config_dict = config_dict
        self.config = init_config_object(self.config_dict)
        self.device = self.config.device
        self.load_checkpoint()
        self.init_state()
        self.data = data_class(self.config)
        self.model_class = model_class
        self.init_model()
        self.init_optimizer()

    def run(self):
        self.evaluation_step()
        self.evaluation_step_log()
        self.sample()
        loader = self.data.train_loader()
        for epoch in range(self.config.epoch):
            self.state.epoch = epoch
            for step, data in enumerate(loader):
                # looks ugly but the logic is very easy to read
                self.state.step = step + 1
                self.do_forward_backward_step(data)
                if self.should_optimize():
                    self.do_optimization()
                    if self.should_evaluate():
                        self.do_evaluation()
                        if self.should_save_checkpoint():
                            self.save_checkpoint()

            self.do_optimization()
            self.do_evaluation()
            if self.should_save_checkpoint():
                self.save_checkpoint()

    @staticmethod
    def iterable_to_device(iterable, device):
        return tuple(x.to(device) for x in iterable)

    def load_checkpoint(self):
        if not self.config.init_from == "checkpoint":
            return
        checkpoint_path = Path("checkpoint") / (
            self.config.experiment_name + ".ckpt"
        )
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

    def init_state(self):
        self.state = SimpleNamespace()
        if self.config.init_from == "checkpoint":
            self.state = self.checkpoint["state"]
            return
        self.state.step = 0
        self.state.epoch = 0
        self.state.optimization_step = 0
        self.state.best_save_metric = -float("inf")

    def init_model(self):
        model = self.model_class(self.config)
        if self.config.init_from == "pretrained":
            model.copy_pretrained()
        elif self.config.init_from == "checkpoint":
            model.load_checkpoint(self.checkpoint["model"])
        self.model = model.to(self.device)
        if self.config.freeze:
            self.model.freeze()

    def init_optimizer(self):
        self.optimizer = self.model.create_optimizer()
        if self.config.init_from == "pretrained":
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])

    def forward_backward_step(self, data):
        with torch.autocast(
            device_type=self.config.device, dtype=torch.bfloat16
        ):
            _, loss = self.model.training_step(data)
        self.state.train_loss = round(loss.item(), 3)
        self.state.loss = loss / self.config.gradient_accumulation_steps
        self.state.loss.backward()

    def handle_save_metric(self):
        self.state.save_metric = -self.state.eval_loss

    def should_optimize(self):
        return (self.state.step) % self.config.gradient_accumulation_steps == 0

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimization_step += 1

    def optimize_step_log(self):
        keys = "mode", "epoch", "optimization_step", "train_loss"
        vals = ["optimization"]
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            vals.append(getattr(self.state, k))
        state_log.log(keys, vals)

    def do_optimization(self):
        self.optimize_step()
        self.optimize_step_log()

    def should_evaluate(self):
        return (self.state.optimization_step) % self.config.eval_interval == 0

    @torch.no_grad()
    def sample(self):
        self.model.eval()
        loader = self.data.valuation_loader()
        for i, data in enumerate(loader):
            data = self.iterable_to_device(data, self.device)
            input_, label = data
            prediction = self.model.sample(input_)
            self.sample_log(input_, label, prediction)
            if i == self.config.n_sample:
                break
        self.model.train()

    def sample_log(self, input_, label, prediction):
        print("input")
        print(self.data.decode(input_))
        print("expected output")
        print(self.data.decode(label))
        print("real output")
        print(self.data.decode(prediction))

    @torch.no_grad()
    def evaluation_step(self):
        self.model.eval()
        loader = self.data.valuation_loader()
        losses = torch.zeros(self.config.eval_iters)
        self.state.eval_predictions = []
        self.state.eval_labels = []
        for i, data in enumerate(loader):
            data = self.iterable_to_device(data, self.device)
            prediction, eval_loss = self.model.evaluation_step(data)
            losses[i] = eval_loss
            *_, label = data
            self.state.eval_labels.append(label)
            self.state.eval_predictions.append(prediction)
            if i == self.config.eval_iters - 1:
                break
        self.state.metric = self.data.get_metrics(
            self.state.eval_labels, self.state.eval_predictions
        )
        self.state.eval_loss = round(losses.mean().item(), 4)
        self.handle_save_metric()
        self.model.train()

    def evaluation_step_log(self):
        keys = [
            "mode",
            "epoch",
            "optimization_step",
            "eval_loss",
            "metric",
        ]
        vals = ["evaluation"]
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            vals.append(getattr(self.state, k))
        state_log.log(keys, vals)

    def do_evaluation(self):
        self.evaluation_step()
        self.evaluation_step_log()
        self.sample()

    def should_save_checkpoint(self):
        if self.state.best_save_metric >= self.state.save_metric:
            return False
        self.state.best_save_metric = self.state.save_metric
        print(
            f"new best metric: {self.state.best_save_metric}, save checkpoint..."
        )
        return True

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state": self.state,
        }

        checkpoint_path = Path("checkpoint") / (
            self.config.experiment_name + ".ckpt"
        )
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def do_forward_backward_step(self, data):
        data = self.iterable_to_device(data, self.device)
        self.forward_backward_step(data)


def run_training(data_class, model_class):
    config_file_path = sys.argv[2]
    config_dict = load_config_dict(config_file_path)
    t = Trainer(config_dict, data_class, model_class)
    t.run()
