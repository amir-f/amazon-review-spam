__author__ = 'Amir'
import logging

logging.basicConfig(level=logging.DEBUG, format='%(process)d\t%(asctime)s:%(levelname)s: %(message)s', datefmt='%H:%M:%S')

from pre_process import crawl_to_graph

DS_DIR = '/home/amir/pyproj/amazon-review-spam/io/same_cat_v2'

graph, membs, prods = crawl_to_graph(ds_dir=DS_DIR)
graph_orig = graph.copy()

import networkx as nx
from os import path
mgraph = nx.read_gexf(path.join(DS_DIR, '%s.gexf' % 'em_unlabeled_mgraph'))


author_product_mapping = {}
for a in mgraph:
    author_product_mapping[a] = [p for p in graph[a]]


from hardEM_gurobi import HardEM

nparts = 4
ll, partition = HardEM.run_EM(author_graph=mgraph, author_product_map=author_product_mapping, nparts=nparts*5, parallel=True, nprocs=4)

for a in mgraph:
    mgraph.node[a]['cLabel'] = int(partition[a])


nx.write_gexf(mgraph, path.join(DS_DIR, '%s.gexf' % 'em_labeled_mgraph'), version='1.2draft', encoding='us-ascii')