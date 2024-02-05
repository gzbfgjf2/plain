import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
import tomli
import json
from collections import namedtuple
from pathlib import Path

n_rjust_width = 15


# def create_config(path):
#     with open(path, "rb") as f:
#         dictionary = tomli.load(f)
#     config = json.loads(
#         json.dumps(dictionary),
#         object_hook=lambda d: namedtuple("Config", d.keys())(*d.values()),
#     )
#     return dictionary, config


def iterable_to_device(iterable, device):
    return tuple(x.to(device) for x in iterable)


def log_format(title, pairs):
    message = title.rjust(n_rjust_width) + ""
    for k, v in pairs:
        message += k.rjust(n_rjust_width) + ": "
        message += str(v).rjust(n_rjust_width) + " "
    return message


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
        self.data = data_class()
        self.init_model(model_class)
        self.optimizer = self.model.create_optimizer()
        self.state = SimpleNamespace()
        self.state.step = 0
        self.state.best_metric = 0

    def init_model(self, model_class):
        # hard code the mapping instead of dynamic, to prevent injection attack
        m = {
            "__init__": "__init__",
            "from_pretrained": "from_pretrained",
            "from_checkpoint": "from_checkpoint",
        }
        if self.config.model_from == "__init__":
            model = model_class(self.config)
        else:
            model = getattr(model_class, m[self.config.model_from])(
                self.config
            )
        model = model.to(self.device)
        self.model = model

    def forward_backward_step(self, data):
        _, loss = self.model.training_step(data)
        self.state.train_loss = round(loss.item(), 3)
        self.state.loss = loss / self.config.gradient_accumulation_steps
        self.state.loss.backward()

    def evaluation_coroutine(self):
        # loader = DataLoader(self.test, batch_size=None, shuffle=False)
        loader = self.data.valuation_loader()
        self.state.eval_predictions = []
        self.state.eval_labels = []
        for data in loader:
            data = iterable_to_device(data, self.device)
            *_, y = data
            prediction = yield data
            # append needs to be below yield, as last item will yield but will
            # not receive prediciton. If not below, len(labels)-1 ==
            # len(prediction)
            self.state.eval_labels.append(y)
            self.state.eval_predictions.append(prediction)
        # if does not yield one more, it will break after
        # last_item = coroutine.send()
        # so we cannot send the last predicition
        yield torch.empty(1)

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def evaluation_step(self):
        self.model.eval()
        evaluation_coroutine = self.evaluation_coroutine()
        losses = torch.zeros(self.config.eval_iters)
        i = 0
        test_data = next(evaluation_coroutine)
        test_data = iterable_to_device(test_data, self.device)
        while True:
            ids, eval_loss = self.model.evaluation_step(test_data)
            losses[i] = eval_loss
            test_data = evaluation_coroutine.send(ids)
            test_data = iterable_to_device(test_data, self.device)
            i += 1
            if i == self.config.eval_iters:
                break
        self.state.eval_metric = self.data.get_metrics(
            self.state.eval_labels, self.state.eval_predictions
        )
        self.state.eval_loss = round(losses.mean().item(), 4)
        self.model.train()

    def should_optimize(self, step):
        return step % self.config.gradient_accumulation_steps == 0

    def should_evaluate(self, step):
        return (
            step
            % (
                self.config.eval_iters
                * self.config.gradient_accumulation_steps
            )
            == 0
        )

    def optimize_step_log(self):
        keys = "epoch", "step", "train_loss"
        title = "[training]"
        pairs = []
        for k in keys:
            if not hasattr(self.state, k):
                continue
            pairs.append([k, getattr(self.state, k)])

        print(log_format(title, pairs))

    def evaluation_step_log(self):
        keys = ["epoch", "step", "train_loss", "eval_loss", "eval_metric"]
        title = "[evaluation]"
        pairs = []
        for k in keys:
            if not hasattr(self.state, k):
                continue
            pairs.append([k, getattr(self.state, k)])
        print(log_format(title, pairs))

    def should_save_checkpoint(self, *args):
        if self.state.best_metric < self.state.eval_metric:
            self.state.best_metric = self.state.eval_metric
            print(f"new best metric: {self.state.best_metric}")
            return True
        return False

    def save_checkpoint(self, *args):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config_dict": self.config_dict,
            "state": self.state,
        }

        checkpoint_path = Path("checkpoint") / (
            self.config.experiment_name + ".ckpt"
        )
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def run(self):
        self.metric = self.evaluation_step()
        self.evaluation_step_log()
        # loader = DataLoader(self.data.train, batch_size=None, shuffle=True)
        loader = self.data.train_loader()
        for epoch in range(self.config.epoch):
            self.state.epoch = epoch
            for step, data in enumerate(loader):
                data = iterable_to_device(data, self.device)
                self.forward_backward_step(data)
                if self.should_optimize(step):
                    self.optimize_step()
                    self.optimize_step_log()
                    self.state.step += 1
                if self.should_evaluate(step):
                    self.evaluation_step()
                    self.evaluation_step_log()
                    if self.should_save_checkpoint():
                        self.save_checkpoint()
