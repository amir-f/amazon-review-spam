from graph_cluster import *

__author__ = 'Amir'


if __name__ == '__main__':
    # parameters
    N_CLUSTERS = (2,2)
    EM_ITER = 30
    EM_REP = 8
#    graph_file_in = 'in.graphml'
#    graph_file_out = 'out.graphml'
    graph_file_in = 'spam_graph.gexf'
    graph_file_out = 'spam_graph_clustered.gexf'

    graph = read_graph_from_file(graph_file_in)
    members = [n for n in graph.nodes() if 'isRealName' in graph.node[n]]
    prods = [n for n in graph.nodes() if 'avail' in graph.node[n]]
    for m in members:
      graph.node[m]['part'] = 1
    for p in prods:
      graph.node[p]['part'] = 2

    graph = graph_cluster_EM(graph, N_CLUSTERS=N_CLUSTERS, EM_ITER=EM_ITER, EM_REP=EM_REP,
                             bipartite=True, use_rev_len=False)
    write_graph_to_file(graph, graph_file_out)
