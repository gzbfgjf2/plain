import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
import tomli
import json
from collections import namedtuple
from pathlib import Path
import os

n_rjust_width = 20


def iterable_to_device(iterable, device):
    return tuple(x.to(device) for x in iterable)


class StateLog:
    def __init__(self):
        self.previous_keys = None

    def log(self, keys, vals):
        vals = "".join([f"{v}".rjust(n_rjust_width) for v in vals])
        if keys == self.previous_keys:
            print(vals)
            return
        self.previous_keys = keys
        keys = "".join([k.rjust(n_rjust_width) for k in keys])
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
        self.state = SimpleNamespace()
        self.state.step = 0
        self.state.epoch = 0
        self.state.optimization_step = 0
        self.state.best_save_metric = -float("inf")
        self.config_dict = config_dict
        self.config = init_config_object(self.config_dict)
        self.device = self.config.device
        self.data = data_class(self.config)
        self.init_model(model_class)
        self.optimizer = self.model.create_optimizer()
        if self.config.model_from == "from_checkpoint":
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])

    def init_model(self, model_class):
        # hard code the mapping instead of dynamic, to prevent injection attack
        m = {
            "__init__": "__init__",
            "from_pretrained": "from_pretrained",
            "from_checkpoint": "from_checkpoint",
        }
        if self.config.model_from == "__init__":
            model = model_class(self.config)
        elif self.config.model_from == "from_checkpoint":
            checkpoint_path = Path("checkpoint") / (
                self.config.experiment_name + ".ckpt"
            )
            self.checkpoint = torch.load(
                checkpoint_path, map_location=self.config.device
            )

            model = model_class.from_checkpoint(self.config, self.checkpoint)

            self.state = self.checkpoint["state"]
            print(
                f"loading checkpoint... \nprevious best metric:\n{self.state.best_save_metric}"
            )
            # assert self.config_dict == checkpoint["config_dict"]
        elif self.config.model_from == "from_pretrained":
            model = getattr(model_class, m[self.config.model_from])(
                self.config
            )
        else:
            assert False, f"{self.config.model_from} not implemented"

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
        self.state.optimization_step += 1

    def evaluation_step(self):
        self.model.eval()
        evaluation_coroutine = self.evaluation_coroutine()
        losses = torch.zeros(self.config.eval_iters)
        i = 0
        test_data = next(evaluation_coroutine)
        test_data = iterable_to_device(test_data, self.device)
        while True:
            ids, logits, eval_loss = self.model.evaluation_step(test_data)
            losses[i] = eval_loss
            test_data = evaluation_coroutine.send(ids)
            test_data = iterable_to_device(test_data, self.device)
            i += 1
            if i == self.config.eval_iters:
                break
        self.state.metric = self.data.get_metrics(
            self.state.eval_labels, self.state.eval_predictions
        )
        self.state.eval_loss = round(losses.mean().item(), 4)
        self.handle_save_metric()

        print_n = 5
        labels = self.state.eval_labels[:print_n]
        predictions = self.state.eval_predictions[:print_n]
        for label, prediction in zip(labels, predictions):
            pretty_tokens(self.data.decode(label))
            pretty_tokens(self.data.decode(prediction))

        self.model.train()

    def handle_save_metric(self):
        self.state.save_metric = -self.state.eval_loss

    def should_optimize(self):
        return (self.state.step) % self.config.gradient_accumulation_steps == 0

    def should_evaluate(self):
        return (self.state.optimization_step) % self.config.eval_interval == 0

    def optimize_step_log(self):
        keys = "mode", "epoch", "optimization_step", "train_loss"
        vals = ["optimization"]
        for k in keys[1:]:
            if not hasattr(self.state, k):
                assert False, f"{k} not in self.state"
            vals.append(getattr(self.state, k))

        state_log.log(keys, vals)

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

    def should_save_checkpoint(self, *args):
        if self.state.best_save_metric >= self.state.save_metric:
            return False
        self.state.best_save_metric = self.state.save_metric
        print(
            f"new best metric: {self.state.best_save_metric}, save checkpoint..."
        )
        return True

    def save_checkpoint(self, *args):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.model.optimizer.state_dict(),
            "config_dict": self.config_dict,
            "state": self.state,
        }

        checkpoint_path = Path("checkpoint") / (
            self.config.experiment_name + ".ckpt"
        )
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def run(self):
        self.evaluation_step()
        self.evaluation_step_log()
        # loader = DataLoader(self.data.train, batch_size=None, shuffle=True)
        loader = self.data.train_loader()
        for epoch in range(self.config.epoch):
            self.state.epoch = epoch
            for step, data in enumerate(loader):
                # looks ugly but the logic is very easy to read
                self.state.step = step + 1
                self.do_step(data)
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

    def do_step(self, data):
        data = iterable_to_device(data, self.device)
        self.forward_backward_step(data)

    def do_optimization(self):
        self.optimize_step()
        self.optimize_step_log()

    def do_evaluation(self):
        self.evaluation_step()
        self.evaluation_step_log()
