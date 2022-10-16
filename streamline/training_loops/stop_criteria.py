from abc import ABC, abstractmethod

import torch

class StopCriterion(ABC):
    
    @abstractmethod
    def is_criterion_met(self, encompassing_training_loop):
        pass
    
    @abstractmethod
    def get_progress_towards_criterion(self, encompassing_training_loop):
        pass
    
    def get_parameter_value_pairs(self):
        return list(self.__dict__.items())
    
    def set_parameter_values(self, parameter_value_pairs):
        for parameter_name, value in parameter_value_pairs:
            setattr(self, parameter_name, value)
    
    
class MaxAccuracyCriterion(StopCriterion):
    
    def __init__(self, max_accuracy):
        self.max_accuracy = max_accuracy
        
    def is_criterion_met(self, encompassing_training_loop):
        return encompassing_training_loop.current_train_accuracy >= self.max_accuracy
    
    def get_progress_towards_criterion(self, encompassing_training_loop):        
        progress_towards_criterion = encompassing_training_loop.current_train_accuracy / self.max_accuracy 
        return progress_towards_criterion
    
    
class MaxEpochCriterion(StopCriterion):
    
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        
    def is_criterion_met(self, encompassing_training_loop):
        return encompassing_training_loop.epoch >= self.max_epoch
    
    def get_progress_towards_criterion(self, encompassing_training_loop):
        progress_towards_criterion = encompassing_training_loop.epoch / self.max_epoch
        return progress_towards_criterion


class MaxPlateauCriterion(StopCriterion):

    def __init__(self, optimizer, max_plateaus):
        self.max_plateaus = max_plateaus
        self.current_plateaus = 0
        self.current_lr = self.get_optim_lr(optimizer)


    def get_optim_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def is_criterion_met(self, encompassing_training_loop):
        
        # Get all the current learning rates in the optimizer.
        optimizer = encompassing_training_loop.optimizer
        new_lr = self.get_optim_lr(optimizer)
        
        # If the lrs have changed, then a plateau occurred
        if self.current_lr != new_lr:
            self.current_plateaus += 1
            self.current_lr = new_lr

        return self.current_plateaus >= self.max_plateaus


    def get_progress_towards_criterion(self, encompassing_training_loop):
        return self.current_plateaus / self.max_plateaus