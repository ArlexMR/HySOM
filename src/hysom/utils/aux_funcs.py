
def split_range(start, end, num_parts):
    if num_parts <= 0:
        raise ValueError("Number of parts must be positive.")
    if start >= end:
        raise ValueError("Start must be less than end.")

    range_length = end - start
    part_size = range_length // num_parts
    remainder = range_length % num_parts

    boundaries = [start]
    for i in range(num_parts):
        next_boundary = boundaries[-1] + part_size + (1 if i < remainder else 0)
        boundaries.append(next_boundary)

    return boundaries

def split_range_auto(start, end, max_parts):
    if start >= end:
        raise ValueError("Start must be less than end.")
    range_length = max(1, end - start)
    max_parts = min(max_parts, range_length)  
    min_parts = max(1, range_length//3)
    best_part_size = float('inf')
    best_num_parts = 0

    for num_parts in range(min_parts, max_parts + 1):
        part_size = range_length / num_parts
        if part_size.is_integer() and part_size < best_part_size:
            best_part_size = part_size
            best_num_parts = num_parts

    if best_num_parts == 0:  # If no equal parts found, use maximum parts within reasonable boundary
        best_num_parts = max_parts

    return split_range(start, end, best_num_parts)