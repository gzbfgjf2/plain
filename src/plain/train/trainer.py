import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader


n_rjust_width = 15


def iterable_to_device(iterable, device):
    return tuple(x.to(device) for x in iterable)


def log_format(title, pairs):
    message = title.rjust(n_rjust_width) + ""
    for k, v in pairs:
        message += k.rjust(n_rjust_width) + ": "
        message += str(v).rjust(n_rjust_width) + " "
    return message


class Trainer:
    def __init__(self, config, data, model):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = self.model.create_optimizer()
        self.state = SimpleNamespace()
        self.state.step = 0
        self.predictions = []
        self.labels = []

    def forward_backward_step(self, data):
        _, loss = self.model.training_step(data)
        self.state.train_loss = round(loss.item(), 3)
        self.state.loss = loss / self.config.gradient_accumulation_steps
        self.state.loss.backward()

    def evaluation_coroutine(self):
        # loader = DataLoader(self.test, batch_size=None, shuffle=False)
        loader = self.data.valuation_loader()
        self.predictions = []
        self.labels = []
        for data in loader:
            data = iterable_to_device(data, self.device)
            *_, y = data
            prediction = yield data
            # append needs to be below yield, as last item will yield but will
            # not receive prediciton. If not below, len(labels)-1 ==
            # len(prediction)
            self.labels.append(y)
            self.predictions.append(prediction)
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
        self.state.metrics = self.data.get_metrics(
            self.labels, self.predictions
        )
        self.state.val_loss = round(losses.mean().item(), 4)
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
        keys = ["epoch", "step", "train_loss", "val_loss", "metrics"]
        title = "[evaluation]"
        pairs = []
        for k in keys:
            if not hasattr(self.state, k):
                continue
            pairs.append([k, getattr(self.state, k)])
        print(log_format(title, pairs))

    def handle_save(self):
        pass

    def should_save_checkpoint(self, *args):
        return False

    def save_checkpoint(self, *args):
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
                if self.should_save_checkpoint(step):
                    self.save_checkpoint()
