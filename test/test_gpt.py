import unittest
from plain.model.gpt import GPT as Model
from plain.train import load_config_dict, init_config_object
import torch


class TestGPT(unittest.TestCase):
    def test_model_initialization(self):
        config_path = "./test/config/test_gpt.toml"
        config_dict = load_config_dict(config_path)
        config = init_config_object(config_dict)
        try:
            Model(config)
        except Exception as e:
            self.fail(f"ModelA initialization failed: {e}")

    def test_model_forward(self):
        config_path = "./test/config/test_gpt.toml"
        config_dict = load_config_dict(config_path)
        config = init_config_object(config_dict)
        try:
            model = Model(config)
            input = torch.randint(0, config.n_vocab, (10, 10))
            print(model.forward)
            model(input)
        except Exception as e:
            self.fail(f"Model initialization failed: {e}")


if __name__ == "__main__":
    unittest.main()
