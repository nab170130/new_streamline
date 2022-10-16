import torch
import unittest

from streamline import models
from torch import nn


class TestModels(unittest.TestCase):
    

    def setUp(self):

        base_num_classes = 1000
        self.model_factory = models.ModelFactory(num_classes=base_num_classes)
        self.img_test_labels = torch.randint(base_num_classes,(5,))
        self.model_names = models.__ALL_MODELS__


    def set_test_tensor(self, model_name):
        if model_name.startswith("mnistnet") or model_name.startswith("deepermnistnet"):
            self.img_test_tensor = torch.randn(5,1,28,28)
        elif model_name.endswith("_1c"):
            self.img_test_tensor = torch.randn(5,1,32,32)
        else:
            self.img_test_tensor = torch.randn(5,3,32,32)


    def test_num_classes(self):
        
        # Try one configuration of classes
        num_classes = 10
        model_factory = models.ModelFactory(num_classes=num_classes)

        for model_name in self.model_names:
            model = model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out = model(self.img_test_tensor)
            self.assertEqual(num_classes, out.shape[1])

        # Try another configuration of classes
        num_classes = 20
        model_factory = models.ModelFactory(num_classes=num_classes)

        for model_name in self.model_names:
            model = model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out = model(self.img_test_tensor)
            self.assertEqual(num_classes, out.shape[1])


    def test_channels(self):
        
        # Try all model variants for 3 channels
        for model_name in self.model_names:
            model = self.model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out = model(self.img_test_tensor)

        # Try all model variants for 1 channel
        for model_name in self.model_names:
            model = self.model_factory.get_model(F"{model_name}_1c")
            self.set_test_tensor(F"{model_name}_1c")
            out = model(self.img_test_tensor)


    def test_last(self):
        
        # Try all last parameters 
        for model_name in self.model_names:
            model = self.model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out, last = model(self.img_test_tensor, last=True)

            # Check that last embeddings have same dimension as get_embedding_dim
            emb_dim = model.get_embedding_dim()
            self.assertEqual(last.shape[1], emb_dim)


    def test_freeze(self):
        
        # For each model, calculate a loss (CE), and compute gradients. If more than one 
        # parameter group changes (e.g., more than just the last linear layer), then the model is not freezing
        # correctly.
        loss_func = nn.CrossEntropyLoss()
        for model_name in self.model_names:

            # Randomly initialize the model
            model = self.model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out = model(self.img_test_tensor, freeze=True)
            loss = loss_func(out, self.img_test_labels)
            loss.backward()

            # Go over the models parameters, checking how many parameter groups have non-zero gradients
            num_weight_params_nonzero = 0
            num_bias_params_nonzero = 0
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                if gradient is not None:
                    if ".weight" in name:
                        num_weight_params_nonzero += 1
                    else:
                        num_bias_params_nonzero += 1
            
            # Check that either weight, bias is zero or 1 and that ONE is at least 1.
            self.assertIn(num_weight_params_nonzero, [0,1])
            self.assertIn(num_bias_params_nonzero, [0,1])
            self.assertGreaterEqual(num_weight_params_nonzero + num_bias_params_nonzero, 1)


    def test_variable_sizes(self):
        
        # Create a tensor of different size
        diff_size_img_test_tensor = torch.randn(5, 3, 64, 64)
        for model_name in self.model_names:
            if model_name.startswith("deepermnistnet") or model_name.startswith("mnistnet"):
                continue
            model = self.model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out = model(diff_size_img_test_tensor)

            # Make sure the model output a tensor with 2-dim shape
            self.assertEqual(len(out.shape), 2)

        # Try a smaller size (like _MNIST)
        diff_size_img_test_tensor = torch.randn(5, 3, 28, 28)
        for model_name in self.model_names:
            if model_name.startswith("deepermnistnet") or model_name.startswith("mnistnet"):
                continue
            model = self.model_factory.get_model(model_name)
            self.set_test_tensor(model_name)
            out = model(diff_size_img_test_tensor)

            # Make sure the model output a tensor with 2-dim shape
            self.assertEqual(len(out.shape), 2)