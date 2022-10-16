from abc import ABC, abstractmethod

class Experiment(ABC):
    
    def __init__(self, comm_lock, comm_sem, parent_pipe, gpu_name, base_dataset_directory, base_exp_directory, db_loc):
        self.parent_pipe = parent_pipe
        self.comm_lock = comm_lock
        self.comm_sem = comm_sem
        self.gpu_name = gpu_name
        self.base_dataset_directory = base_dataset_directory
        self.base_exp_directory = base_exp_directory
        self.db_loc = db_loc
        
        self.validation_fraction = 0.2   
        
    def notify_parent(self, message):
        
        self.comm_lock.acquire()            # Prevent other worker processes from writing into the pipe
        self.parent_pipe.send(message)      # Send the message
        self.comm_sem.acquire()             # Wait until the manager process has read from the pipe
        self.comm_lock.release()            # Let other worker processes through
        
    @abstractmethod
    def experiment(self):
        pass