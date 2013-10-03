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

logging.basicConfig(level=logging.WARN, format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%H:%M:%S')


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
    prt = np.zeros(n, dtype=np.int8)
    if m <= n:
        return _gen_partition(n, m, prt)
    else:
        return []


def _gen_partition(n, m, prt):
    """
    Given the pre allocated partition array and result list,
    generates all possible partitioning of n elements into m partitions
    """
    if n == m:
        prt[:n] = range(n)
        yield tuple(prt)
        return
    if m == 1:
        prt[:n] = np.zeros(n)
        yield tuple(prt)
        return
    # last element on its own partition
    prt[n - 1] = m - 1
    for p in _gen_partition(n - 1, m - 1, prt):
        yield p
    # last element is  in any of the m partitions
    for i in range(m):
        prt[n - 1] = i
        for p in _gen_partition(n - 1, m, prt):
            yield p


def gen_synthetic_bicluster_graph(N):
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


def gen_synthetic_graph(N, nc):
    graph = nx.Graph(name='synthezied author graph')
    cluster_sizes = [int(round(cs)) for cs in dirichlet([7] * nc) * N]
    ph_s = dirichlet([1] * nc)
    pr_s = dirichlet([1] * nc)
    pv_s = dirichlet([1] * nc)
    SIGMA = 0.6
    TAU = 0.9
    AVG_PER_CLASS_PROD = 5
    mus = normal(loc=5.5, scale=3, size=nc)
    all_products = range(nc * AVG_PER_CLASS_PROD)
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
    # components = nx.connected_components(graph)
    # largest_component_i = np.argmax([len(c) for c in components])
    # largest_component = set(components[largest_component_i])
    # graph.remove_nodes_from([n for n in graph if n not in largest_component])
    # generate author_prod_map
    for n in graph:
        ci = graph.node[n]['acluster']
        nprods = randint(1, len(all_products)/2)
        author_prod_map[n] = list(np.nonzero(multinomial(nprods, pi_s[ci]))[0])

    return graph, author_prod_map, cluster_sizes


def test_hard_EM(N, nparts, write_labeled_graph=True, parallel=True):
    graph, author_prod_map, _ = gen_synthetic_graph(N, nparts)
    ll, partition = HardEM.run_EM(author_graph=graph, author_product_map=author_prod_map, nparts=nparts, parallel=parallel)

    print 'best loglikelihood: %s' % ll
    print partition.values()
    for n in partition:
        graph.node[n]['cLabel'] = int(partition[n])
    if write_labeled_graph:
        nx.write_graphml(graph, '/home/amir/amazon-spam-review/io/synthetic_graph_labeled.graphml')
    return graph


def rand_index(prt, ref_prt):
    n = len(prt)
    assert n == len(ref_prt)
    t = 0       # No. correct clustering
    for i1, i2 in itertools.combinations(range(len(prt)), 2):
        if (prt[i1] == prt[i2] and ref_prt[i1] == ref_prt[i2]) or (prt[i1] != prt[i2] and ref_prt[i1] != ref_prt[i2]):
            t += 1
    return float(t) / (n*(n-1)/2)


def stirling2(n, k):
    if n == k:
        return 1
    if k == 1:
        return 1
    if k == 0:
        return 0
    if k > n:
        return 0
    return k*stirling2(n-1, k) + stirling2(n-1, k-1)


def em_ll_map(prt):
    em = HardEM(author_graph=ex_ll_graph, author_product_map=ex_ll_author_prod_map, nparts=ex_ll_nparts, init_partition=prt)
    return prt, em.log_likelihood(), rand_index(prt, ex_ll_ref_prt)


def exhaustive_ll(N, nparts, parallel=True):
    global ex_ll_graph, ex_ll_nparts, ex_ll_author_prod_map, ex_ll_ref_prt
    ex_ll_graph, ex_ll_author_prod_map, cluster_sizes = gen_synthetic_graph(N, nparts)
    N = sum(cluster_sizes)      # sum of cluster sizes is close to N but does not always match
    ex_ll_nparts = nparts
    ex_ll_graph, ex_ll_author_prod_map = HardEM._preprocess_graph_and_map(ex_ll_graph, ex_ll_author_prod_map)
    # reference partitioning
    ex_ll_ref_prt = []
    for i in range(len(cluster_sizes)):
        ex_ll_ref_prt.extend([i]*cluster_sizes[i])
    ex_ll_ref_prt = tuple(ex_ll_ref_prt)
    # all possible partitioning of at most `nparts` partitions
    partitions = itertools.chain(*[gen_partition(N, nparts_i) for nparts_i in range(1, nparts + 1)])
    logging.info('Processing %d partitions' % sum(stirling2(N, nparts_i) for nparts_i in range(1, nparts + 1)))
    if parallel:
        p = Pool()
        v = p.imap(em_ll_map, partitions)
        p.close(); p.join()
    else:
        v = itertools.imap(em_ll_map, partitions)
    v = list(v)     # since v is a generator, keeps them in a list so reading from it won't consume it
    # find the logl for the presumed correct partitioning
    ref_ll = 0
    for vv in v:
        if vv[0] == ex_ll_ref_prt:
            ref_ll = vv[1]
            break
    else:
        logging.error('The correct partitioning was not found')
    # keep only one from set of permutations with the same loglikelihood
    # v_dict = {ll: prt for prt, ll in v}
    # v = v_dict.items()
    # v.sort(key=lambda tup: tup[0], reverse=True)
    # for i in range(0, min(10, len(v))):
    #     print '#%d\t%s' % (i, v[i])
    # print '##\t%s' % ((ref_ll, ex_ll_ref_prt),)
    return v, cluster_sizes, ex_ll_graph


def test_real_graph(nparts):
    logging.info('Reading author collab graph')
    author_graph = nx.read_graphml('/home/amir/az/io/spam/mgraph2.gexf')
    author_graph.name = 'author graph'
    logging.info('Reading the full author product graph')
    full_graph = nx.read_graphml('/home/amir/az/io/spam/spam_graph.graphml')
    full_graph.name = 'full graph'

    proper_author_graph = author_graph.subgraph([a for a in author_graph if 'revLen' in author_graph.node[a]
                                                                            and 'hlpful_fav_unfav' in author_graph.node[a]
                                                                            and 'vrf_prchs_fav_unfav' in author_graph.node[a]])
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
    logging.debug('Running EM')
    ll, partition = HardEM.run_EM(proper_author_graph, author_product_mapping, nparts=nparts, parallel=True)
    print 'best loglikelihood: %s' % ll
    for n in partition:
        author_graph.node[n]['cLabel'] = int(partition[n])
    nx.write_gexf(author_graph, '/home/amir/az/io/spam/spam_graph_mgraph_sage_labeled.gexf')


if __name__ == '__main__':
    # pydevd.settrace('192.168.11.212', port=4187, stdoutToServer=True, stderrToServer=True)
    exhaustive_ll(10, 2, True)
    # test_hard_EM(50, 10, write_labeled_graph=False, parallel=False)
    # test_real_graph(nparts=8)
    # test_real_graph_2()