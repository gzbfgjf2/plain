import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader


# Data:
#     self.data.train
#     self.data.evaluation
#     self.data.coroutine()
#     self.data.get_metric
#


def iterable_to_device(iterable, device):
    return (x.to(device) for x in iterable)


class Trainer:
    def __init__(self, config, data, model):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = self.model.create_optimizer()
        self.state = SimpleNamespace()
        self.state.step = 0

    def forward_backward_step(self, data):
        _, loss = self.model.training_step(data)
        self.state.train_loss = round(loss.item(), 3)
        self.state.loss = loss / self.config.gradient_accumulation_steps
        self.state.loss.backward()

    def evaluation_step(self):
        self.model.eval()
        evaluation_coroutine = self.data.evaluation_coroutine()
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
        self.state.metrics = self.data.get_metrics()
        self.state.val_loss = round(losses.mean().item(), 4)
        self.model.train()

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

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
        states = {}
        keys = ["train_loss", "epoch", "step"]
        for k in keys:
            states[k] = getattr(self.state, k)
        print(states)

    def evaluation_step_log(self):
        states = {}
        keys = ["train_loss", "val_loss", "epoch", "step", "metrics"]
        for k in keys:
            if hasattr(self.state, k):
                states[k] = getattr(self.state, k)
        print("evaluation")
        print(states)

    def handle_save(self):
        pass

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
