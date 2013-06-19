import os
from collections import deque
import csv
import networkx as nx
from networkx.algorithms import bipartite
from scrapy.crawler import Crawler
import re
from datetime import datetime
import time
import logging
import itertools

__author__ = 'Amir'

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
_realNumberRE = re.compile(r'\d+(?:,\d+)?(?:\.\d+)?')
_DATE_FORMAT = {'spam': r'%d-%b-%y', 'legit': r'%B %d, %Y'}


def _dataset_path(dataset):
    return r'%s/io/%s' % (Crawler.settings['PROJECT_PATH'], Crawler.settings['DATA_SET'])


def _read_num_tuple(tupleStr):
    all = _realNumberRE.findall(tupleStr)
    return [float(nstr) for nstr in all]


def _read_scrapy_csv(memberFile, prodFile, revFile, seedFile, dataset, **kwarg):
    members = {}
    prods = {}
    resGraph = nx.Graph()

    logging.info('Reading reviews')
    logging.debug('Review file: ' + revFile)
    # Reviews
    numeric_date = kwarg.get('numeric_date', True)
    helpful_ratio_total = kwarg.get('helpful_ratio_total', True)
    with open(revFile, 'r') as readFile:
        reader = csv.DictReader(readFile)
        for rev in reader:
            revCopy = rev.copy()
            del revCopy['productId']
            del revCopy['memberId']
            if numeric_date:
                dt = datetime.strptime(rev['date'], _DATE_FORMAT[dataset])
                revCopy['date'] = time.mktime(dt.timetuple())
            if rev['helpful']:
                revCopy['helpful'] = _read_num_tuple(revCopy['helpful'])
                if helpful_ratio_total:
                    helpful = revCopy['helpful']
                    del revCopy['helpful']
                    revCopy['helpful_ratio'] = float(helpful[0]) / float(helpful[1])
                    revCopy['helpful_total'] = helpful[1]
            else:
                del revCopy['helpful']
            revCopy['starRating'] = _read_num_tuple(revCopy['starRating'])[0]
            revCopy['reviewTxt'] = rev['reviewTxt'].decode('windows-1252', 'ignore')
            if rev['title']:
                revCopy['title'] = rev['title'].decode('windows-1252', 'ignore')
            else:
                del revCopy['title']
            revCopy['verifiedPurchase'] = revCopy['verifiedPurchase'] == 'TRUE'
            resGraph.add_edge(rev['productId'], rev['memberId'], revCopy)

    logging.info('Reading members')
    logging.debug('Member file: ' + memberFile)
    # Members
    with open(memberFile, 'r') as readFile:
        reader = csv.DictReader(readFile)
        for m in reader:
            mc = m.copy()
            del mc['id']
            if m['location']:
                mc['location'] = m['location'].decode('windows-1252', 'ignore')
            else:
                del mc['location']
            mc['fullname'] = m['fullname'].decode('windows-1252', 'ignore')
            mc['isRealName'] = m['isRealName'] == 'TRUE'
            members[m['id']] = mc
            resGraph.node[m['id']] = mc

    logging.info('Reading products')
    logging.debug('Product file: ' + prodFile)
    # Products
    numeric_price = kwarg.get('numeric_price', True)
    with open(prodFile, 'r') as readFile:
        reader = csv.DictReader(readFile)
        for p in reader:
            pc = p.copy()
            del pc['id']
            pc['name'] = p['name'].decode('windows-1252', 'ignore')
            if p['cat']:
                pc['cat'] = p['cat'].decode('windows-1252', 'ignore')
            else:
                del pc['cat']
            pc['avail'] = p['avail'] == 'TRUE'
            if p['price']:
                if numeric_price:
                    #a dirty fix to make price values an int
                    m = _realNumberRE.search(p['price'])
                    if m:
                        pc['price'] = float(m.group(0).replace(',', ''))       # remove , digit separator
                    else:
                        del pc['price']
                else:
                    pc['price'] = p['price']
            else:
                del pc['price']
            prods[p['id']] = pc
            resGraph.node[p['id']] = pc

    logging.info('Reading seeds')
    logging.debug('Seed file: ' + seedFile)
    seed = set()
    with open(seedFile, 'r') as readFile:
        reader = csv.DictReader(readFile)
        for s in reader:
            seed.add(s['ID'])

    return resGraph, members, prods, seed


def read_scrapy_csv(**kwarg):
    """
    Reads scrapy CSV output file and load the graph into memory.
    """

    datasetPath = _dataset_path(kwarg.get('dataset', Crawler.settings['DATA_SET']))
    return _read_scrapy_csv(
        memberFile=kwarg.get('memberFile', '%s/%s' % (datasetPath, Crawler.settings['OUTPUT_FILE_MEMBER'])),
        prodFile=kwarg.get('prodFile', '%s/%s' % (datasetPath, Crawler.settings['OUTPUT_FILE_PRODUCT'])),
        revFile=kwarg.get('revFile', '%s/%s' % (datasetPath, Crawler.settings['OUTPUT_FILE_REVIEW'])),
        seedFile=kwarg.get('seedFile', '%s/%s' % (datasetPath, Crawler.settings['SPIDER_SEED_FILENAME'])),
        **kwarg
    )


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


def _create_if_needed(revFileOut):
    dirName = os.path.dirname(revFileOut)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    return revFileOut


def read_scrapy_write_gephi(**kwarg):
    dataset = kwarg.get('dataset', Crawler.settings['DATA_SET'])
    logging.debug('Dataset is ' + dataset)

    graph, membs, prods, seeds = read_scrapy_csv(numeric_date=True, numeric_price=True,
                                                 helpful_ratio_total=True, **kwarg)
    #setting paths
    graph_file_out = r'%s/%s.graphml' % (
        Crawler.settings['DATA_SET_DIR'], kwarg.get('graph_file_out', Crawler.settings['OUTPUT_FILE_GRAPH']))
    graph = scrapy_to_gephi(graph=graph, seeds=seeds)

    logging.info('Writing to GraphML')
    nx.write_graphml(graph, graph_file_out)


def scrapy_to_gephi(graph, seeds):
    """
    Prepares Scrapy output for Gephi
    """

    logging.info('Preparing Scrapy output for Gephi')
    graph = analysis(graph=graph, seed=seeds)

    logging.info('Preprocessing graph')
    for edge in graph.edges():
        u, v = edge
        rev = graph[u][v]
        rev['Type'] = 'Undirected'
        graph[u][v] = rev

    return graph


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


def analysis2(members, prods, graph):
    print bipartite.density(graph, members)
    print bipartite.density(graph, prods)
    return bipartite.clustering(graph, members)


def m_projection(graph_orig, members, prods):
    logging.info('Projecting the graph')

    graph = graph_orig.copy()
    #considering only favorable edges
    graph.remove_edges_from([e for e in graph.edges(data=True) if e[2]['starRating'] < 4])
    assert set(graph) == (set(members) | set(prods))

    WINDOW = 60 * 60 * 24 * 3
    mg = nx.Graph()
    mg.add_nodes_from(members)

    prod_names = dict()
    for p in prods:
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

    logging.debug('Normalizing edge weights: meet/min')
    for e in mg.edges():
        u, v = e
        norm = min(len(nx.neighbors(graph, u)), len(nx.neighbors(graph, v)))
        mg.add_edge(u, v, weight=float(mg[u][v]['weight']) / float(norm), denom=norm)
        # remove isolated nodes
    degrees = mg.degree()
    mg.remove_nodes_from([n for n in mg if degrees[n] == 0])
    # adding original graph metadata on nodes
    for m in mg:
        mg.node[m] = graph_orig.node[m]
    logging.debug(r'Projected Nodes = %d, Projected Edges = %d' % (mg.order(), len(mg.edges())))

    return mg, prod_names


if __name__ == '__main__':
    graph = nx.read_gexf('%s/%s/amazon_full_favorable.gexf' % (Crawler.settings['PROJECT_PATH'], 'spam'))
    members = [n for n in graph if 'isRealName' in graph.node[n]]
    prods = [n for n in graph if 'avail' in graph.node[n]]
    mg = m_projection(graph, members, prods)
    nx.write_gexf(mg, '%s/%s/memberProjection_full.gexf' % (Crawler.settings['PROJECT_PATH'], 'spam'))

__all__ = ['read_scrapy_csv', 'scrapy_to_gephi', 'analysis', 'read_scrapy_write_gephi', 'm_projection']