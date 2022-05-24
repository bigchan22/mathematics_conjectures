def get_sinks(D):
    return (node for node, out_dg in D.out_degree() if out_dg == 0)

def shortest_path_lengths(D):
    sinks = get_sinks(D)
    shortest_lengths = dict.fromkeys(D.nodes, float('inf'))
    for sink in sinks:
        for node, length in nx.shortest_path_length(D, target=sink).items():
            if shortest_lengths[node] > length:
                shortest_lengths[node] = length
    return shortest_lengths

def longest_path_length_to_target(D, target):
    dist = dict.fromkeys(D.nodes, -float('inf'))
    dist[target] = 0
    topo_order = reversed(list(nx.topological_sort(D)))
    for v in topo_order:
        for u in D.predecessors(v):
            if dist[u] < dist[v] + 1:
                dist[u] = dist[v] + 1
    return dist

def longest_path_lengths(D):
    sinks = get_sinks(D)
    longest_lengths = dict.fromkeys(D.nodes, -float('inf'))
    for sink in sinks:
        for node, length in longest_path_length_to_target(D, target=sink).items():
            if longest_lengths[node] < length:
                longest_lengths[node] = length
    return longest_lengths
