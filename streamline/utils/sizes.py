def calculate_new_coreset_sizes(coreset_sizes, buffer_capacity):
    
    number_of_current_coresets = len(coreset_sizes)
    coreset_sizes_with_index = [(coreset_size, i) for i, coreset_size in enumerate(coreset_sizes)]
    sorted_coreset_sizes_with_index = sorted(coreset_sizes_with_index, key=lambda x:x[0])
    sorted_coreset_sizes = [x[0] for x in sorted_coreset_sizes_with_index]
    sorted_to_original_mapping = [x[1] for x in sorted_coreset_sizes_with_index]
    
    cumulative_new_size = 0
    
    for i in range(number_of_current_coresets):
        
        num_remaining_coresets = number_of_current_coresets - i
        if i != 0:
            shared_size_of_all_but_i_smallest_coresets = sorted_coreset_sizes[i] - sorted_coreset_sizes[i - 1]
        else:
            shared_size_of_all_but_i_smallest_coresets = sorted_coreset_sizes[i]
        prospective_cumulative_new_size = cumulative_new_size + num_remaining_coresets * shared_size_of_all_but_i_smallest_coresets
        
        if prospective_cumulative_new_size > buffer_capacity:
            
            if i != 0:
                last_shared_coreset_size_before_overfilling = sorted_coreset_sizes[i - 1]
            else:
                last_shared_coreset_size_before_overfilling = 0
            size_to_distribute_to_remaining_coresets = (buffer_capacity - cumulative_new_size) // num_remaining_coresets
            remainder = buffer_capacity - num_remaining_coresets * size_to_distribute_to_remaining_coresets - cumulative_new_size
        
            final_coreset_sizes = []
            for j in range(number_of_current_coresets):
                final_coreset_size = min(sorted_coreset_sizes[j], last_shared_coreset_size_before_overfilling + size_to_distribute_to_remaining_coresets)
                if remainder > 0 and final_coreset_size < sorted_coreset_sizes[j]:
                    final_coreset_size += 1
                    remainder -= 1
                final_coreset_sizes.append(final_coreset_size)
            final_coreset_sizes_in_original_order = [0 for x in range(number_of_current_coresets)]
            for j in range(number_of_current_coresets):
                original_index = sorted_to_original_mapping[j]
                final_coreset_sizes_in_original_order[original_index] = final_coreset_sizes[j]
        
            return final_coreset_sizes_in_original_order
            
        else:
            cumulative_new_size = prospective_cumulative_new_size
    
    return coreset_sizes