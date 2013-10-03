from collections import deque
import networkx as nx
from networkx.algorithms import bipartite
import re
import logging
import itertools
from scipy.integrate import dblquad
from scipy.stats import gaussian_kde

__author__ = 'Amir'

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
_realNumberRE = re.compile(r'\d+(?:,\d+)?(?:\.\d+)?')
_DATE_FORMAT = {'spam': r'%d-%b-%y', 'legit': r'%B %d, %Y'}


def _bfs_depth(graph, seed):
    depth = {}
    for s in seed:
        depth[s] = 0
    q = deque(seed)
    while len(q) > 0:
        n = q.popleft()
        for adjN in graph.neighbors_iter(n):
            if adjN not in depth:
                depth[adjN] = depth[n] + 1
                q.append(adjN)
    return depth


def analysis(graph, seed, calc_nrd=True, calc_ncc=True, calc_depth=True):
    """
    Computes and returns a number of statistics on the graph
    """

    logging.info('Computing Statistics')

    depth, nrd, ncc = {}, {}, {}
    if calc_depth:
        depth = _bfs_depth(graph, seed)
    if calc_nrd:
        nrd = bipartite.node_redundancy(graph)
    if calc_ncc:
        ncc = bipartite.clustering(graph, mode='min')

    for id in graph.nodes():
        node = graph.node[id]
        if calc_depth:
            node['depth'] = depth[id]
        if calc_nrd:
            node['nrd'] = nrd[id]
        if calc_ncc:
            node['ncc'] = ncc[id]
        graph.node[id] = node

    return graph


def bipartite_analysis(members, prods, graph):
    print bipartite.density(graph, members)
    print bipartite.density(graph, prods)
    return bipartite.clustering(graph, members)


WINDOW = 60 * 60 * 24 * 7
SIGNF_LEVEL = 0.05
EPS = 1e-6
MIN_TS_LEN = 50


def m_projection(graph_orig, members, prods, full_graph):
    logging.info('Projecting the graph on members')

    graph = graph_orig.subgraph(graph_orig.nodes())
    #considering only favorable edges
    graph.remove_edges_from([e for e in graph.edges(data=True) if e[2]['starRating'] < 4])
    assert set(graph) == (set(members) | set(prods))

    mg = nx.Graph()
    mg.add_nodes_from(members)

    prods_len = float(len(prods))
    last_pctg = 0
    prod_names = dict()
    for p_i, p in enumerate(prods):
        # first check whether two favorable reviews within a WINDOW is significant enough (p-value < 0.5)
        ts = [e['date'] for e in full_graph[p].values()]
        # In order for gkde to work, there should be more than one point value
        if len(ts) >= MIN_TS_LEN and min(ts) < max(ts):
            gkde = gaussian_kde(ts)
            p_value, err = dblquad(lambda u, v: gkde(u)*gkde(v), min(ts) - WINDOW/2.0, max(ts) + WINDOW/2.0,
                                   lambda v: v - WINDOW/2.0, lambda v: v + WINDOW/2.0)
            if p_value - EPS >= SIGNF_LEVEL and err < EPS:
                continue
        for m1, m2 in itertools.combinations(nx.neighbors(graph, p), 2):
            # order m1,m2 so the key (m1,m2) for prod_names works regardless of edge direction
            if m1 > m2:
                m1, m2 = m2, m1
                #assert m1 in members and m2 in members
            if abs(graph[p][m1]['date'] - graph[p][m2]['date']) < WINDOW:
                if mg.has_edge(m1, m2):
                    c = mg[m1][m2]['weight']
                else:
                    c = 0
                    prod_names[(m1, m2)] = []
                prod_names[(m1, m2)].append(p)
                mg.add_edge(m1, m2, weight=c + 1)
        pctg = int(p_i/prods_len*100)
        if pctg % 10 == 0 and pctg > last_pctg:
            last_pctg = pctg
            logging.info('%d%% Done' % pctg)

    logging.debug('Normalizing edge weights: meet/max')
    for e in mg.edges():
        u, v = e
        if mg[u][v]['weight'] <= 1:
            mg.remove_edge(u, v)
            del prod_names[(min(u, v), max(u, v))]
        else:
            norm = max(len(nx.neighbors(graph, u)), len(nx.neighbors(graph, v)))
            mg.add_edge(u, v, weight=float(mg[u][v]['weight']) / norm, denom=norm)
    # remove isolated nodes
    degrees = mg.degree()
    mg.remove_nodes_from([n for n in mg if degrees[n] == 0])
    # adding original graph metadata on nodes
    for m in mg:
        mg.node[m] = graph_orig.node[m]
    logging.debug(r'Projected Nodes = %d, Projected Edges = %d' % (mg.order(), len(mg.edges())))

    return mg, prod_names


def p_projection(graph_orig, members, prods):
    logging.info('Projecting the graph on products')

    graph = graph_orig.copy()
    #considering only favorable edges
    graph.remove_edges_from([e for e in graph.edges(data=True) if e[2]['starRating'] < 4])
    assert set(graph) == (set(members) | set(prods))

    pg = nx.Graph()
    pg.add_nodes_from(prods)

    memb_names = dict()
    for m in members:
        for p1, p2 in itertools.combinations(nx.neighbors(graph, m), 2):
            # order m1,m2 so the key (m1,m2) for memb_names works regardless of edge direction
            if p1 > p2:
                p1, p2 = p2, p1
                #assert m1 in members and m2 in members
            if abs(graph[m][p1]['date'] - graph[m][p2]['date']) < WINDOW:
                if pg.has_edge(p1, p2):
                    c = pg[p1][p2]['weight']
                else:
                    c = 0
                    memb_names[(p1, p2)] = []
                memb_names[(p1, p2)].append(m)
                pg.add_edge(p1, p2, weight=c + 1)

    logging.debug('Normalizing edge weights: meet/min')
    for e in pg.edges():
        u, v = e
        norm = min(len(nx.neighbors(graph, u)), len(nx.neighbors(graph, v)))
        pg.add_edge(u, v, weight=float(pg[u][v]['weight']) / float(norm), denom=norm)
        # remove isolated nodes
    degrees = pg.degree()
    pg.remove_nodes_from([n for n in pg if degrees[n] == 0])
    # adding original graph metadata on nodes
    for m in pg:
        pg.node[m] = graph_orig.node[m]
    logging.debug(r'Projected Nodes = %d, Projected Edges = %d' % (pg.order(), len(pg.edges())))

    return pg, memb_names


__all__ = ['analysis', 'bipartite_analysis', 'm_projection', 'p_projection']
