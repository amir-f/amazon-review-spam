__author__ = 'Amir'

from hardEM_gurobi import HardEM
import networkx as nx
import logging
from multiprocessing import Pool
import numpy as np
import itertools
import random
import pydevd
from numpy.random import dirichlet, normal, binomial, multinomial
from random import randint

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%H:%M:%S')


def test_graph1():
    graph = nx.Graph(name='author graph')
    graph.add_node(1, hlpful_fav_unfav=True, isRealName=True, revLen=10)
    graph.add_node(2, hlpful_fav_unfav=True, isRealName=True, revLen=12)
    graph.add_node(3, hlpful_fav_unfav=False, isRealName=False, revLen=9)
    graph.add_node(4, hlpful_fav_unfav=False, isRealName=True, revLen=30000)
    graph.add_node(5, hlpful_fav_unfav=True, isRealName=False, revLen=40000)
    graph.add_node(6, hlpful_fav_unfav=False, isRealName=False, revLen=40000)

    graph.add_edge(1, 2, weight=3)
    graph.add_edge(1, 3, weight=3)
    graph.add_edge(2, 3, weight=3)
    graph.add_edge(3, 4, weight=1)
    graph.add_edge(4, 5, weight=3)
    graph.add_edge(5, 6, weight=3)
    graph.add_edge(4, 6, weight=3)
    return graph


def test_graph2(w8=0.3):
    graph = nx.Graph(name='author graph')
    graph.add_node(1, hlpful_fav_unfav=True, isRealName=True, revLen=100)
    graph.add_node(2, hlpful_fav_unfav=True, isRealName=True, revLen=120)
    graph.add_node(3, hlpful_fav_unfav=False, isRealName=False, revLen=95)
    graph.add_node(4, hlpful_fav_unfav=False, isRealName=True, revLen=500)
    graph.add_node(5, hlpful_fav_unfav=True, isRealName=False, revLen=600)
    graph.add_node(6, hlpful_fav_unfav=False, isRealName=False, revLen=850)

    graph.add_edge(1, 2, weight=w8)
    graph.add_edge(1, 3, weight=w8)
    graph.add_edge(2, 3, weight=w8)
    graph.add_edge(3, 4, weight=w8)
    graph.add_edge(4, 5, weight=w8)
    graph.add_edge(5, 6, weight=w8)
    graph.add_edge(4, 6, weight=w8)
    return graph


def gen_partition(n, m):
    """
    Generates all possible partitioning of n elements into m partitions. The size of the output will be
    stirling number of second kind of {n, m}
    """
    prt_list = []
    prt = np.zeros(n, dtype=np.int8)
    if m <= n:
        _gen_partition(n, m, prt, prt_list)
    return prt_list


def _gen_partition(n, m, prt, prt_list):
    """
    Given the pre allocated partition array and result list,
    generates all possible partitioning of n elements into m partitions
    """
    if n == m:
        prt[:n] = range(n)
        prt_list.append(tuple(prt))
        return
    if m == 1:
        prt[:n] = np.zeros(n)
        prt_list.append(tuple(prt))
        return
    # last element on its own partition
    prt[n - 1] = m - 1
    _gen_partition(n - 1, m - 1, prt, prt_list)
    # last element is  in any of the m partitions
    for i in range(m):
        prt[n - 1] = i
        _gen_partition(n - 1, m, prt, prt_list)


def gen_test_graph(N):
    logging.info('Generating test graph')
    graph = nx.Graph(name='author graph')
    p_h = 0.8
    p_r = 0.8
    p_v = 0.1
    mu_1, sigma_1 = 5, 1
    mu_2, sigma_2 = 7, 2
    strong_weight = 0.9
    weak_weight = 0.1
    clique_density = 0.4
    author_prod_map = {}
    for a in range(0, N//2):
        graph.add_node(a, hlpful_fav_unfav=binomial(1, p_h) == 1, isRealName=binomial(1, p_r) == 1,
                       vrf_prchs_fav_unfav=binomial(1, p_v) == 1, revLen=normal(mu_1, sigma_1))
        author_prod_map[a] = np.random.randint(N//2, size=N)

    for a in range(N//2, N):
        graph.add_node(a, hlpful_fav_unfav=binomial(1, 1 - p_h) == 1, isRealName=binomial(1, 1 - p_r) == 1,
                       vrf_prchs_fav_unfav=binomial(1, p_v) == 1, revLen=normal(mu_2, sigma_2))
        author_prod_map[a] = np.random.randint(N//2, N, size=N)

    clique1 = list(itertools.combinations(xrange(0, N//2), 2))
    for a, b in random.sample(clique1, int(len(clique1)*clique_density)):
        graph.add_edge(a, b, weight=strong_weight, denom=5)
    graph.add_edge(N//2 - 1, N//2, weight=weak_weight, denom=5)
    clique2 = list(itertools.combinations(xrange(N//2, N), 2))
    for a, b in random.sample(clique2, int(len(clique2)*clique_density)):
        graph.add_edge(a, b, weight=strong_weight, denom=5)
    # add non edges with zero weight
    # non_edge_edges = list(itertools.product(xrange(0, N//2), xrange(N//2, N)))
    # for a, b in random.sample(non_edge_edges, int(len(non_edge_edges)*clique_density)):
    #     graph.add_edge(a, b, weight=non_edge_weight, denom=5)
    return graph, author_prod_map


def create_synthetic_graph(N, nc):
    graph = nx.Graph(name='synthezied author graph')
    cluster_sizes = [int(cs) for cs in dirichlet([7] * nc) * N]
    ph_s = dirichlet([1] * nc)
    pr_s = dirichlet([1] * nc)
    pv_s = dirichlet([1] * nc)
    SIGMA = 0.6
    TAU = 0.99
    mus = normal(loc=5.5, scale=3, size=nc)
    all_products = range(nc * 50)
    pi_s = []
    for ci in range(nc):
        pi_s.append(dirichlet([0.5] * len(all_products)))
    author_prod_map = {}

    # generate nodes
    for ci in range(nc):
        for ni in range(cluster_sizes[ci]):
            graph.add_node(len(graph), acluster=ci, revLen=normal(loc=mus[ci], scale=SIGMA),
                           isRealName=binomial(1, pr_s[ci]) == 1, hlpful_fav_unfav=binomial(1, ph_s[ci]) == 1,
                           vrf_prchs_fav_unfav=binomial(1, pv_s[ci]) == 1)
    # generate edges
    for a, b in itertools.combinations(graph.nodes(), 2):
        if not binomial(1, min(15.0/len(graph), 1.0)):
            continue
        if graph.node[a]['acluster'] == graph.node[b]['acluster']:
            if binomial(1, TAU):
                graph.add_edge(a, b, weight=np.clip(normal(1, scale=0.25), 0, 1), denom=5)
        else:
            if binomial(1, 1 - TAU):
                graph.add_edge(a, b, weight=np.clip(normal(0.5, scale=0.25), 0, 1), denom=5)
    # keep only the largest component
    components = nx.connected_components(graph)
    largest_component_i = np.argmax([len(c) for c in components])
    largest_component = set(components[largest_component_i])
    graph.remove_nodes_from([n for n in graph if n not in largest_component])
    # generate author_prod_map
    for n in graph:
        ci = graph.node[n]['acluster']
        nprods = randint(1, len(all_products)/2)
        author_prod_map[n] = list(np.nonzero(multinomial(nprods, pi_s[ci]))[0])

    return graph, author_prod_map


def test_hard_EM(N, nparts, write_labeled_graph=True, parallel=True):
    graph, author_prod_map = gen_test_graph(N)
    ll, partition = HardEM.run_EM(author_graph=graph, author_product_map=author_prod_map, nparts=nparts, parallel=parallel)

    print 'best loglikelihood: %s' % ll
    print partition.values()
    for n in partition:
        graph.node[n]['cLabel'] = int(partition[n])
    if write_labeled_graph:
        nx.write_graphml(graph, '/home/amir/az/io/spam/synthetic_graph_sage_labeled.graphml')


def em_ll_map(prt):
    em = HardEM(author_graph=ex_ll_graph, author_product_map=ex_ll_author_prod_map, nparts=ex_ll_nparts, init_partition=prt)
    return prt, em.log_likelihood()


def exhaustive_ll(N, nparts, parallel=True):
    global ex_ll_graph, ex_ll_nparts, ex_ll_author_prod_map
    ex_ll_graph, ex_ll_author_prod_map = gen_test_graph(N)
    ex_ll_nparts = nparts
    # all possible partitioning of at most `nparts` partitions
    partitions = []
    for nparts_i in range(1, nparts + 1):
        partitions.extend(gen_partition(N, nparts_i))
    logging.info('Processing %d partitions' % len(partitions))
    if parallel:
        p = Pool()
        v = p.map(em_ll_map, partitions)
        p.close(); p.join()
    else:
        v = map(em_ll_map, partitions)
    # find the logl for the presumed correct partitioning
    ref_prt = tuple([0] * (N//2) + [1] * (N - N//2))
    ref_ll = 0
    for vv in v:
        if vv[0] == ref_prt:
            ref_ll = vv[1]
            break
    else:
        logging.error('The correct partitioning was not found')
    # keep only one from set of permutations with the same loglikelihood
    v_dict = {ll: prt for prt, ll in v}
    v = v_dict.items()
    v.sort(key=lambda tup: tup[0], reverse=True)
    for i in range(0, min(10, len(v))):
        print '#%d\t%s' % (i, v[i])
    print '##\t%s' % ((ref_ll, ref_prt),)
    return v


def test_real_graph(nparts):
    MIN_CC_SIZE = 10        # Nodes belonging to connected components smaller than this are discarded
    logging.info('Reading author collab graph')
    author_graph = nx.read_graphml('/home/amir/az/io/spam/spam_mgraph_augmented.graphml')
    author_graph.name = 'author graph'
    logging.info('Reading the full author product graph')
    full_graph = nx.read_graphml('/home/amir/az/io/spam/spam_graph.graphml')
    full_graph.name = 'full graph'

    logging.info('Removing nodes which do not have all the features')
    proper_author_graph = author_graph.subgraph([a for a in author_graph if 'revLen' in author_graph.node[a]
                                                and 'hlpful_fav_unfav' in author_graph.node[a]
                                                and 'vrf_prchs_fav_unfav' in author_graph.node[a]])
    logging.info('Keeping only nodes which belong to large connected components')
    ccs = nx.connected_components(proper_author_graph)
    ccs = filter(lambda cc: len(cc) >= MIN_CC_SIZE, ccs)
    proper_author_graph = proper_author_graph.subgraph(itertools.chain(*ccs))
    # features = {'revLen': 0.0, 'hlpful_fav_unfav': False, 'vrf_prchs_fav_unfav': False}
    # for a in author_graph:
    #     for feat, def_val in features.items():
    #         if feat not in author_graph.node[a]:
    #             author_graph.node[a][feat] = def_val

    # sub sample proper_author_graph
    # proper_author_graph.remove_edges_from(random.sample(proper_author_graph.edges(), 2*proper_author_graph.size()/3))
    # degree = proper_author_graph.degree()
    # proper_author_graph.remove_nodes_from([n for n in proper_author_graph if degree[n] == 0])
    # author to the product reviewed by him mapping
    logging.debug('forming the product mapping')
    author_product_mapping = {}
    for a in proper_author_graph:
        author_product_mapping[a] = [p for p in full_graph[a] if 'starRating' in full_graph[a][p] and
                                                                 full_graph[a][p]['starRating'] >= 4]
    logging.info('Running EM')
    ll, partition = HardEM.run_EM(proper_author_graph, author_product_mapping, nparts=nparts, parallel=True)
    print 'best loglikelihood: %s' % ll
    for n in partition:
        author_graph.node[n]['cLabel'] = int(partition[n])
    output_filename = 'spam_graph_mgraph_labeled.gexf'
    logging.info('Writing the clusters into the graph and saving the file into %s'%output_filename)
    nx.write_gexf(author_graph, '/home/amir/az/io/spam/%s'%output_filename)

if __name__ == '__main__':
    # pydevd.settrace('192.168.11.227', port=4187, stdoutToServer=True, stderrToServer=True)
    # exhaustive_ll(16, 2, True)
    # test_hard_EM(50, 10, write_labeled_graph=False, parallel=False)
    test_real_graph(nparts=8)