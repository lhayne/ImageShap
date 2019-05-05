def topological_sort(tensor):
    visited = set()
    top_sort = []
    def helper(t):
        if (t in visited) or t.is_leaf:
            pass
        else:
            visited.add(t)
            for p in t.grad_fn.parents:
                helper(p)
            top_sort.append(t)
    helper(tensor)
    return top_sort
