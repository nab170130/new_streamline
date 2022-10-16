from torch.nn.functional import normalize

import torch
import torch.linalg

class LazyGreedyMatroidPartitionOptimizer():
    
    def __init__(self, submodular_function, ground_set_partitions, verbose=False):
        self.submodular_function = submodular_function
        self.ground_set_partitions = ground_set_partitions
        self.verbose = verbose
        self.num_elem = submodular_function.n
        

    def heapify(self, heap_array, priority_function):
        
        num_elements_to_heapify = len(heap_array)
        for index in range(num_elements_to_heapify-1,-1,-1):
            self.percolate_down(index, heap_array, priority_function)


    def percolate_down(self, index, heap_array, priority_function):
        
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        num_elements = len(heap_array)
        
        if left_child_index >= num_elements:
            return
        elif right_child_index >= num_elements:
            left_child_priority = priority_function(heap_array, left_child_index)
            parent_priority = priority_function(heap_array, index)
            if left_child_priority > parent_priority:
                swap = heap_array[index]
                heap_array[index] = heap_array[left_child_index]
                heap_array[left_child_index] = swap
        else:
            left_child_priority = priority_function(heap_array, left_child_index)
            right_child_priority = priority_function(heap_array, right_child_index)
            parent_priority = priority_function(heap_array, index)
            if left_child_priority >= right_child_priority:
                if left_child_priority >= parent_priority:
                    swap = heap_array[index]
                    heap_array[index] = heap_array[left_child_index]
                    heap_array[left_child_index] = swap
                    self.percolate_down(left_child_index, heap_array, priority_function)
            else:
                if right_child_priority >= parent_priority:
                    swap = heap_array[index]
                    heap_array[index] = heap_array[right_child_index]
                    heap_array[right_child_index] = swap
                    self.percolate_down(right_child_index, heap_array, priority_function)
        

    def percolate_up(self, index, heap_array, priority_function):
        
        if index == 0:
            return
        
        parent_index = (index - 1) // 2
        
        priority = priority_function(heap_array, index)
        parent_priority = priority_function(heap_array, parent_index)
        if priority >= parent_priority:
            swap = heap_array[index]
            heap_array[index] = heap_array[parent_index]
            heap_array[parent_index] = swap
            self.percolate_up(parent_index, heap_array, priority_function)
        

    def peek(self, heap_array):
        
        return heap_array[0]
    

    def get_top(self, heap_array, priority_function):
        
        to_return_element = heap_array[0]
        perc_down_element = heap_array[-1]
        heap_array[0] = perc_down_element
        del heap_array[-1]
        self.percolate_down(0, heap_array, priority_function)
        return to_return_element
    

    def insert(self, heap_array, element, priority_function):
        
        insert_element_index = len(heap_array)
        heap_array.append(element)
        self.percolate_up(insert_element_index, heap_array, priority_function)
        

    def maximize(self, partition_constraints, cardinality_constraint):
        
        if len(partition_constraints) != len(self.ground_set_partitions):
            raise ValueError("Partition constraints do not align for provided partition!")
            
        ground_set_size = self.num_elem
        current_partition_counts = [0 for x in range(len(self.ground_set_partitions))]
        current_subset = set()
        
        # Initialize the heap, which stores (index_of_ground_set, priority [or marginal_gain]) tuples
        heap = [(i,self.submodular_function.marginalGain(current_subset,i)) for i in range(ground_set_size)]
        priority_function = lambda x,i : x[i][1]
        self.heapify(heap, priority_function)
        
        # Use variant of Lazy Greedy to maximize w/ partition matroid
        while len(heap) > 0:
            
            # Pop the first element. If it is an element that violates the partition constraints, pop again.
            keep_popping = True
            while keep_popping:
                if len(heap) == 0:
                    return list(current_subset)
                
                potentially_violating_element = self.get_top(heap, priority_function)
                potentially_violating_element_index = potentially_violating_element[0]
                partition_number = -1
                for partition_num in range(len(self.ground_set_partitions)):
                    if potentially_violating_element_index in self.ground_set_partitions[partition_num]:
                        partition_number = partition_num
                        break
                
                # Keep popping if this element is not in the partition
                keep_popping = current_partition_counts[partition_number] >= partition_constraints[partition_number]
                element_to_test = potentially_violating_element
            
            # Re-evaluate the marginal gain of the popped element
            element_to_test_index = element_to_test[0]
            new_marg_gain = self.submodular_function.marginalGain(current_subset, element_to_test_index)
            
            # Try inserting this element back into the heap
            reinsert_element = (element_to_test_index, new_marg_gain)
            self.insert(heap, reinsert_element, priority_function)
            
            # Peek the top.
            new_top_element = self.peek(heap)
            new_top_element_index = new_top_element[0]
            
            # Is the top element the same that was popped? If so, this element 
            # is the one with highest marginal gain, so add it. Otherwise, 
            # repeat the process!
            if new_top_element_index == element_to_test_index:
                self.get_top(heap, priority_function)
                current_subset.add(new_top_element_index)
                current_partition_counts[partition_number] += 1

            # If the current subset matches/exceeds our cardinality constraint, exit.
            if len(current_subset) >= cardinality_constraint:
                break

        # All elements must've been added. In any case, return current_subset.
        return list(current_subset)