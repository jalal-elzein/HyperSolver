
def read_set_cover_instance(file_path):
    """Reads a set cover instance:
       First line: <num_elements> <num_subsets>
       Next lines: each line is a list of element IDs belonging to that subset.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split()
        if len(header) < 2:
            raise ValueError("Invalid header format for set cover: need #elements #subsets")
        num_elements = int(header[0])
        num_subsets = int(header[1])
        elements = set()
        subsets = []
        for i in range(num_subsets):
            if i+1 >= len(lines):
                raise ValueError(f"Missing subset data for subset {i}")
            subset = list(map(int, lines[i+1].strip().split()))
            if not subset:
                print(f"Warning: empty subset at line {i+1}")
            subset = [e for e in subset if 0 < e <= num_elements]
            subsets.append(subset)
            elements.update(subset)
        if len(elements) != num_elements:
            print(f"Warning: #elements mismatch: header says {num_elements}, unique elements={len(elements)}")
        if len(subsets) != num_subsets:
            print(f"Warning: #subsets mismatch: header says {num_subsets}, actual {len(subsets)}")
        return subsets, list(range(1, num_elements+1)), header
    except Exception as e:
        print(f"Error reading instance file: {str(e)}")
        raise

def read_subset_sum_instance(file_path):
    """Reads a subset sum instance and creates meaningful hypergraph structure:
       First line: <num_items> <target_sum>
       Next lines: each line has one item weight.
       
       Creates weight-class based hypergraph structure:
       - Items become nodes (elements) 
       - Global hyperedge connects all items (representing sum constraint)
       - Weight-class hyperedges group items by weight ranges (light/medium/heavy)
       - Provides meaningful structure for neural network learning
    """
    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines()]
    hdr = lines[0].split()
    if len(hdr) < 2:
        raise ValueError("Invalid first line: need <num_items> <target_sum>")
    num_items = int(hdr[0])
    if len(hdr) == 2:
        target_sum = float(hdr[1])
        weights = []
        for i in range(num_items):
            w = float(lines[i+1].strip())
            weights.append(w)
        
        max_weight = max(weights)
        if max_weight > 1e6:
            scale_factor = 10000.0 / max_weight
            weights = [w * scale_factor for w in weights]
            target_sum = target_sum * scale_factor
            print(f"[Subset Sum] Applied numerical scaling: factor={scale_factor:.2e}, new_target={target_sum:.0f}")
        else:
            print(f"[Subset Sum] No scaling needed: max_weight={max_weight:.0f}")
        
        elements = list(range(1, num_items + 1))
        
        min_weight = min(weights)
        max_weight = max(weights)
        weight_range = max_weight - min_weight
        
        if weight_range < 1e-6:
            light_threshold = min_weight
            heavy_threshold = max_weight
        else:
            light_threshold = min_weight + 0.33 * weight_range
            heavy_threshold = min_weight + 0.67 * weight_range
        
        subsets = []
        
        subsets.append(elements.copy())
        
        light_items = [i+1 for i, w in enumerate(weights) if w <= light_threshold]
        if light_items:
            subsets.append(light_items)
        
        medium_items = [i+1 for i, w in enumerate(weights) if light_threshold < w <= heavy_threshold]
        if medium_items:
            subsets.append(medium_items)
            
        heavy_items = [i+1 for i, w in enumerate(weights) if w > heavy_threshold]
        if heavy_items:
            subsets.append(heavy_items)
        
        high_value_threshold = target_sum * 0.5
        high_value_items = [i+1 for i, w in enumerate(weights) if w >= high_value_threshold]
        if high_value_items and len(high_value_items) >= 2:
            subsets.append(high_value_items)
        
        combinable_threshold = target_sum * 0.25
        combinable_items = [i+1 for i, w in enumerate(weights) if w <= combinable_threshold]
        if combinable_items and len(combinable_items) >= 3:
            subsets.append(combinable_items)
        
        num_hyperedges = len(subsets)
        out_header = [str(num_items), str(num_hyperedges), str(target_sum)]
        return subsets, elements, out_header, weights
    else:
        raise ValueError("Legacy format not supported for subset sum. Use: <num_items> <target_sum>")

def read_hypermaxcut_instance(file_path):
    """
    Reads a hypergraph maxcut instance.
    First line: <num_nodes> <num_hyperedges>
    Next lines: each line is a hyperedge (list of node IDs).
    Node IDs may be 0-indexed (0..num_nodes-1) or 1-indexed (1..num_nodes);
    the convention is auto-detected from whether a 0 ID appears.
    """
    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines()]
    hdr = lines[0].split()
    if len(hdr) < 2:
        raise ValueError("Need <num_nodes> <num_edges> in first line for hypermaxcut")
    num_nodes = int(hdr[0])
    num_hypered = int(hdr[1])
    hyperedges = []
    for i in range(num_hypered):
        row = list(map(int, lines[i+1].split()))
        hyperedges.append(row)
    header = [str(num_nodes), str(num_hypered)]
    zero_indexed = any(0 in row for row in hyperedges)
    elements = list(range(0, num_nodes)) if zero_indexed else list(range(1, num_nodes+1))
    return hyperedges, elements, header

def read_hitting_set_instance(file_path):
    """
    Reads a hitting set instance.
    First line: <num_sets> <num_elements>
    Next lines: each line is a set (list of element IDs).
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    if len(header) < 2:
        raise ValueError("Invalid header format for hitting set: need #sets #elements")
    num_sets = int(header[0])
    num_elements = int(header[1])
    subsets = []
    for i in range(num_sets):
        if i+1 >= len(lines):
            raise ValueError(f"Missing set data for set {i}")
        s = list(map(int, lines[i+1].strip().split()))
        if not s:
            print(f"Warning: empty set at line {i+1}")
        s = [e for e in s if 0 < e <= num_elements]
        subsets.append(s)
    return subsets, list(range(1, num_elements+1)), header

def generate_incidence_matrix(subsets, elements):
    """
    Generates a dense incidence matrix.
    For set cover: shape (#elements, #subsets).
    For hitting set, you call transpose afterward, etc.
    """
    import torch
    num_elements = len(elements)
    num_subsets = len(subsets)
    threshold_dense = 1_000_000
    if num_elements * num_subsets > threshold_dense:
        row_idx = []
        col_idx = []
        elem_to_idx = { e: i for i,e in enumerate(elements) }
        for j, s in enumerate(subsets):
            for el in s:
                i = elem_to_idx[el]
                row_idx.append(i)
                col_idx.append(j)
        indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
        vals = torch.ones(len(row_idx), dtype=torch.float32)
        spm = torch.sparse_coo_tensor(indices, vals, (num_elements, num_subsets))
        return spm.coalesce().to_dense()
    else:
        mat = torch.zeros(num_elements, num_subsets, dtype=torch.float32)
        elem_to_idx = { e: i for i,e in enumerate(elements) }
        for j, s in enumerate(subsets):
            for el in s:
                i = elem_to_idx[el]
                mat[i, j] = 1.0
        return mat
