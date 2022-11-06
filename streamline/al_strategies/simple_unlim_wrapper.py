class SimpleUnlimitedMemoryWrapperStrategy:

    def __init__(self, wrapped_strategy):
        self.wrapped_strategy   = wrapped_strategy
        self.args               = wrapped_strategy.args


    def get_oracle_task_identity(self):

        # We can get the task identity by using the get_task_number_and_index_in_task() method on any of the idx in the
        # unlabeled dataset. This is because all the incoming data in the unlabeled dataset belongs to one task.
        task_number, _ = self.wrapped_strategy.unlabeled_dataset.get_task_number_and_index_in_task(0)
        return task_number


    def select(self, budget):
        
        # Select using the wrapped strategy
        selected_idx = self.wrapped_strategy.select(budget)

        # Map the selected idx to their partitions. Go ahead and use the given task identity.
        task_identity = self.get_oracle_task_identity()
        selected_unlabeled_idx_partitioned = [[] for x in range(len(self.wrapped_strategy.labeled_dataset.task_idx_partitions))]
        selected_unlabeled_idx_partitioned[task_identity].extend(selected_idx)
        self.args = self.wrapped_strategy.args

        return selected_unlabeled_idx_partitioned