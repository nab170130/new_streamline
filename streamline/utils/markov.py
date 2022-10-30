import time
import torch
import numpy as np

class MarkovChain:
    
    def __init__(self, connection_matrix):
        
        # Element (i,j) refers to the probability that the state transitions 
        # to state j from state i. Hence, each row must sum to 1.
        self._check_matrix(connection_matrix)
        self.connection_matrix = connection_matrix
    

    def _check_matrix(self, connection_matrix):
        
        if len(connection_matrix.shape) != 2:
            raise ValueError("Markov transition matrix is not square")
            
        if connection_matrix.shape[0] != connection_matrix.shape[1]:
            raise ValueError("Markov transition matrix is not square")
            
        for row in connection_matrix:
            if not torch.isclose(torch.sum(row), torch.tensor(1.)):
                raise ValueError("Markov transition matrix probabilities not well defined")
            for element in row:
                if element < 0 or element > 1:
                    raise ValueError("Markov transition matrix probabilities not well defined")
                    

    def sample_sequence(self, length, current_state=0):
        
        generated_sequence = []
        for iteration in range(length):
            generated_sequence.append(current_state)
            transition_probs = self.connection_matrix[current_state]
            generated_prob = torch.rand(1).item()
            for next_state, transition_prob in enumerate(transition_probs):
                generated_prob -= transition_prob.item()
                if generated_prob <= 0:
                    current_state = next_state
                    break
        return generated_sequence


def sample_rare_access_chain(num_tasks, num_rounds, rare_task_number = 2):

    rare_task_number = 2
    rarity = 0.05

    # Seeding
    torch.manual_seed(40)
    np.random.seed(40)

    # Get task arrival pattern. Keep sampling until a sequence with at least one rare task arrival is found.
    rare_access_transition_matrix = torch.ones(num_tasks, num_tasks) * (1 - rarity) / (num_tasks - 1)
    rare_access_transition_matrix[:,rare_task_number] = rarity
    markov_chain = MarkovChain(rare_access_transition_matrix)
    initial_start_state = np.random.randint(num_tasks)
    found_sequence_with_rare_arrival = False
    while not found_sequence_with_rare_arrival:
        task_arrival_pattern = markov_chain.sample_sequence(num_rounds, initial_start_state)
        found_sequence_with_rare_arrival = rare_task_number in task_arrival_pattern[1:]
    
    # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
    new_seed = time.time_ns() % 1000000
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)

    return task_arrival_pattern


def sample_random_access_chain(num_tasks, num_rounds):
    
    # Seeding
    torch.manual_seed(40)
    np.random.seed(40)

    # Get task arrival pattern
    random_access_transition_matrix = torch.ones(num_tasks, num_tasks) / num_tasks
    markov_chain = MarkovChain(random_access_transition_matrix)
    initial_start_state = np.random.randint(num_tasks)
    task_arrival_pattern = markov_chain.sample_sequence(num_rounds, initial_start_state)

    # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
    new_seed = time.time_ns() % 1000000
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)

    return task_arrival_pattern


def sample_sequential_access_chain(num_tasks, num_rounds):

    # Seeding
    torch.manual_seed(40)
    np.random.seed(40)

    # Get task arrival pattern
    sequential_access_transition_matrix = []
    for row_number in range(num_tasks):
        row = [0. for x in range(num_tasks)]
        row[(row_number + 1) % num_tasks] = 1.
        sequential_access_transition_matrix.append(row)
    sequential_access_transition_matrix = torch.tensor(sequential_access_transition_matrix)
    markov_chain = MarkovChain(sequential_access_transition_matrix)
    initial_start_state = np.random.randint(num_tasks)
    task_arrival_pattern = markov_chain.sample_sequence(num_rounds, initial_start_state)

    # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
    new_seed = time.time_ns() % 1000000
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)

    return task_arrival_pattern