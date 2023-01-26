from .stop_criteria import MaxAccuracyCriterion, MaxEpochCriterion, MaxPlateauCriterion
from abc import ABC, abstractmethod
from mmdet.apis import train_detector
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

import torch
import torch.optim as optim

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class AddIndexDataset(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data, label, index
    
    def __len__(self):
        return len(self.wrapped_dataset)

class TrainingLoop(ABC):
    
    def __init__(self, name, training_dataset, train_transform, net, optimizer, lr_sched, loss_function, stop_criteria, per_epoch_callback, args, validation_dataset=None):
        
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

        if 'batch_size' not in args:
            self.batch_size = 1
        else:
            self.batch_size = args['batch_size']
            
        self.name = name
        
        self.train_transform = train_transform
        training_dataset.transform = train_transform
        
        self.training_dataset = AddIndexDataset(training_dataset)
        self.training_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        if validation_dataset is not None:
            self.validation_dataset = AddIndexDataset(validation_dataset)
            self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.net = net
        
        self.epoch = 0
        self.optimizer = optimizer
        self.lr_sched = lr_sched
        self.loss_function = loss_function
        self.stop_criteria = stop_criteria
        self.per_epoch_callback = per_epoch_callback

    @abstractmethod 
    def _train(self):
        pass
    
    @abstractmethod
    def do_callback(self):
        pass
    
    def should_continue_training(self):
        
        if self.epoch < 0:
            return False

        is_one_criterion_met = False
        
        for criterion in self.stop_criteria:
            is_this_criterion_met = criterion.is_criterion_met(self)
            is_one_criterion_met = is_one_criterion_met or is_this_criterion_met
    
        return not is_one_criterion_met
    
    def from_checkpoint(self, epoch, model_state_dict, opt_state_dict, lr_state_dict):

        self.epoch = epoch
        
        self.net.load_state_dict(model_state_dict)
        self.net = self.net.to(self.device)

        self.optimizer.load_state_dict(opt_state_dict)

        if self.lr_sched is not None:
            self.lr_sched.load_state_dict(lr_state_dict)

    def train(self):

        # Reset the model parameters if this is the very first epoch
        if self.epoch == 0:
            self.net.reset()
            
        self.net = self.net.to(device=self.device)        

        while self.should_continue_training(): 
            
            # Train for an epoch
            self._train()

            # Advance the learning rate schedule
            if self.lr_sched is not None:
                self.lr_sched.step()
            
            # Advance the epoch
            self.epoch += 1

            self.do_callback()
            
        # Return the model
        return self.net
    
    
    def extract_training_details(self):
        
        training_loop_name = self.name
        
        # Get the state dictionaries of the optimizer and lr scheduler (if any)
        opt_state_dict = self.optimizer.state_dict()
        if hasattr(self, "lr_sched"):
            lr_state_dict = self.lr_sched.state_dict()
        else:
            lr_state_dict = None
        
        # Get the stop criteria of the training loop
        stop_criteria = self.stop_criteria
        stop_criteria_as_items = []
        for stop_criterion in stop_criteria:
            stop_criterion_name = type(stop_criterion).__name__
            stop_criterion_pname_value_pairs = stop_criterion.get_parameter_value_pairs()
            to_extend_list = [(stop_criterion_name, pname, value) for pname, value in stop_criterion_pname_value_pairs]
            stop_criteria_as_items.extend(to_extend_list)
        
        # Get the train augmentation
        train_augmentation = self.train_transform

        # Return the current epoch if not done training; otherwise, return -1
        return_epoch = self.epoch if self.should_continue_training() else -1
        
        return self.net, training_loop_name, opt_state_dict, lr_state_dict, stop_criteria_as_items, train_augmentation, return_epoch
    

class ClassificationTrainingLoop(TrainingLoop):

    def __init__(self, name, training_dataset, train_transform, net, optimizer, lr_sched, loss_function, stop_criteria, per_epoch_callback, args):

        super(ClassificationTrainingLoop, self).__init__(name, training_dataset, train_transform, net, optimizer, lr_sched, loss_function, stop_criteria, per_epoch_callback, args, validation_dataset=None)
        self.current_train_accuracy = 0.


    def _train(self):
        
        self.net.train()
        acc_final = 0.

        for batch_id, (x, y, idxs) in enumerate(self.training_loader):
            x, y = x.to(device=self.device), y.to(device=self.device)

            self.optimizer.zero_grad()
            out = self.net(x)
            loss = self.loss_function(out, y.long())
            acc_final += torch.sum((torch.max(out,1)[1] == y).float()).item()
            loss.backward()

            self.optimizer.step()
            
        # Set the current training accuracy
        self.current_train_accuracy = acc_final / len(self.training_dataset)
        

    def do_callback(self):
        
        # Determine which kind of stopping criteria are present.
        criteria_types = [type(criterion) for criterion in self.stop_criteria]
        
        # Favor reporting accuracy progress versus epoch progress
        favored_criterion_type = MaxAccuracyCriterion if MaxAccuracyCriterion in criteria_types else MaxEpochCriterion
        if MaxAccuracyCriterion in criteria_types:
            favored_criterion_type = MaxAccuracyCriterion
        elif MaxPlateauCriterion in criteria_types:
            favored_criterion_type = MaxPlateauCriterion
        else:
            favored_criterion_type = MaxEpochCriterion
        
        for criterion in self.stop_criteria:
            if type(criterion) == favored_criterion_type:
                round_progress = criterion.get_progress_towards_criterion(self)
                break
        
        self.per_epoch_callback(self, round_progress, self.current_train_accuracy)


class ClassificationTrainingLoopVal(ClassificationTrainingLoop):

    def __init__(self, name, training_dataset, train_transform, net, optimizer, lr_sched, loss_function, stop_criteria, per_epoch_callback, args, validation_dataset=None):

        super(ClassificationTrainingLoop, self).__init__(name, training_dataset, train_transform, net, optimizer, lr_sched, loss_function, stop_criteria, per_epoch_callback, args, validation_dataset)
        self.current_train_accuracy = 0.


    def _val(self):

        self.net.eval()
        total_loss = 0.

        # Calculate the total loss. We could set the reduction to sum, but it helps reduce code complexity in other locations
        # if we just weigh by the size of the batch.
        with torch.no_grad():
            for batch_id, (x, y, idxs) in enumerate(self.training_loader):
                batch_size = x.shape[0]
                x, y = x.to(device=self.device), y.to(device=self.device)
                out = self.net(x)
                loss = self.loss_function(out, y.long()) * batch_size
                total_loss += loss
            
        # Set the current validation loss
        self.current_validation_loss = total_loss 


    def train(self):
        
        # Do exactly what the base version does; however, additionally calculate validation loss for LR advancement
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        # Reset the model parameters if this is the very first epoch
        if self.epoch == 0:
            self.net = self.net.apply(weight_reset)
            
        self.net = self.net.to(device=self.device)        

        while self.should_continue_training(): 
            
            # Train for an epoch
            self._train()
            self._val()

            # Advance the learning rate schedule
            if self.lr_sched is not None:
                if type(self.lr_sched) == optim.lr_scheduler.ReduceLROnPlateau:
                    self.lr_sched.step(self.current_validation_loss)
                else:
                    self.lr_sched.step()
            
            # Advance the epoch
            self.epoch += 1

            self.do_callback()
            
        # Return the model
        return self.net

        
class DetectionTrainingLoop:

    def __init__(self, train_dataset, model, training_config):
        self.train_dataset      = train_dataset
        self.model              = model
        self.training_config    = training_config 

    
    def train(self):
        train_detector(self.model, [self.train_dataset], self.training_config, validate=False)


class TrainingLoopFactory:
    
    def __init__(self, training_dataset, base_transform, model, per_epoch_callback, args, validation_dataset = None):
        self.training_dataset = training_dataset
        self.base_transform = base_transform
        self.model = model
        self.per_epoch_callback = per_epoch_callback
        self.args = args
        self.validation_dataset = validation_dataset
    
    def get_training_loop(self, training_loop_name):
        
        if training_loop_name == "cross_entropy_sgd_rand_flips":
            
            max_epoch = 500
            max_accuracy_stopping_criterion = MaxAccuracyCriterion(max_accuracy=0.99)
            max_epoch_stopping_criterion = MaxEpochCriterion(max_epoch=max_epoch)
            stopping_criteria = [max_accuracy_stopping_criterion, max_epoch_stopping_criterion]
            
            optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
            
            loss_function = nn.CrossEntropyLoss()

            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), self.base_transform])
            
            training_loop = ClassificationTrainingLoop(training_loop_name, self.training_dataset, train_transform, self.model, optimizer, lr_sched, 
                                                        loss_function, stopping_criteria, self.per_epoch_callback, self.args)

        elif training_loop_name == "cross_entropy_sgd_rand_crops_flips":

            max_epoch = 500
            max_accuracy_stopping_criterion = MaxAccuracyCriterion(max_accuracy=0.99)
            max_epoch_stopping_criterion = MaxEpochCriterion(max_epoch=max_epoch)
            stopping_criteria = [max_accuracy_stopping_criterion, max_epoch_stopping_criterion]
            
            optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
            
            loss_function = nn.CrossEntropyLoss()

            train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), self.base_transform])
            
            training_loop = ClassificationTrainingLoop(training_loop_name, self.training_dataset, train_transform, self.model, optimizer, lr_sched, 
                                                       loss_function, stopping_criteria, self.per_epoch_callback, self.args)

        elif training_loop_name == "cross_entropy_sgd":

            max_epoch = 500
            max_accuracy_stopping_criterion = MaxAccuracyCriterion(max_accuracy=0.99)
            max_epoch_stopping_criterion = MaxEpochCriterion(max_epoch=max_epoch)
            stopping_criteria = [max_accuracy_stopping_criterion, max_epoch_stopping_criterion]
            
            optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
            
            loss_function = nn.CrossEntropyLoss()

            train_transform = self.base_transform
            
            training_loop = ClassificationTrainingLoop(training_loop_name, self.training_dataset, train_transform, self.model, optimizer, lr_sched, 
                                                       loss_function, stopping_criteria, self.per_epoch_callback, self.args)

        elif training_loop_name == "cross_entropy_sgd_rand_crops_flips_val":

            optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)
            
            max_epoch = 500
            validation_stopping_criterion = MaxPlateauCriterion(optimizer, max_plateaus=4)
            max_epoch_stopping_criterion = MaxEpochCriterion(max_epoch=max_epoch)
            stopping_criteria = [validation_stopping_criterion, max_epoch_stopping_criterion]

            loss_function = nn.CrossEntropyLoss()

            train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), self.base_transform])
            
            training_loop = ClassificationTrainingLoopVal(training_loop_name, self.training_dataset, train_transform, self.model, optimizer, lr_sched, 
                                                       loss_function, stopping_criteria, self.per_epoch_callback, self.args, self.validation_dataset)

        elif training_loop_name == "obj_det_train":

            # We want something that simply calls train_detector() once train() is called on the training loop.
            training_loop = DetectionTrainingLoop(self.training_dataset, self.model, self.args)

        return training_loop