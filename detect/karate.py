import graph_cluster
import networkx as nx


__author__ = 'Amir'


if __name__ == '__main__':
    # parameters
    N_CLUSTERS = 2
    EM_ITER = 10
    EM_REP = 6
    graph_file = 'karate.graphml'

    # read the graph
    graph = nx.read_graphml(graph_file)
    adj_mat_r = nx.to_scipy_sparse_matrix(graph, nodelist=graph.nodes(), format='csr')
    adj_mat_c = nx.to_scipy_sparse_matrix(graph, nodelist=graph.nodes(), format='csc')
    ground_truth = [graph.node[n]['faction']-1 for n in graph.nodes()]

    ghat, accuracy = graph_cluster.graph_cluster_evaluate(adj_lists=(adj_mat_r, adj_mat_c),
        C=N_CLUSTERS, EM_ITER=EM_ITER, EM_REP=EM_REP, ground_truth=ground_truth)

    print 'Best Clustering: \n' + str(ghat)
    print 'Rand Measure: ' + str(accuracy)