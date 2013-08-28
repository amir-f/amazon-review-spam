import json
from datetime import datetime
import networkx as nx
import logging
from scraper import settings
from os import path
from scraper.items import Review, Member, Product
import cPickle as pickle

__author__ = 'Amir'
FMT = '%B %d, %Y'
REF_T = datetime(1970, 1, 1)
MEMB_ID_F = 'member_ids'


def convert(val):
    if isinstance(val, dict):
        return {convert(key): convert(value) for key, value in val.iteritems()}
    elif isinstance(val, list):
        return [convert(element) for element in val]
    elif isinstance(val, unicode):
        return val.encode('ascii', 'ignore')
    else:
        return val


def crawl_to_graph(ds_dir=settings.DATA_SET_DIR, edge_metadata=True, node_metadata=True):
    """
    Reads the crawled Amazon data and forms a networkx graph

    :param ds_dir:  data set path
    :param node_metadata:   Whether node metadata should also be put in the graph
    :param edge_metadata:   Whether edge metadata should also be put in the graph
    """
    review_graph = nx.Graph()

    logging.info('Started converting scrapy output to nx.Graph')
    logging.info('Reading review relations')
    membs, prods = set(), set()
    with open(path.join(ds_dir, '%s.json' % Review().export_filename), 'r') as revs_f:
        for l in revs_f:
            d = convert(json.loads(l))
            try:
                d['date'] = (datetime.strptime(d['date'], FMT) - REF_T).total_seconds()
            except ValueError:
                logging.error('Incorrect date stamp for %s' % d['id'])
            if 'helpful' in d:
                h = d['helpful']
                d['helpfulRatio'], d['helpfulTotal'] = float(h[0])/h[1], h[1]
                del d['helpful']
            if 'memberId' in d:
                m_id = d['memberId']
                del d['memberId']
                membs.add(m_id)
            else:
                m_id = d['id']
            if 'productId' in d:
                p_id = d['productId']
                del d['productId']
                prods.add(p_id)
            else:
                p_id = d['id']
            if edge_metadata:
                review_graph.add_edge(m_id, p_id, **d)
            else:
                review_graph.add_edge(m_id, p_id)
    if node_metadata:
        # read member and product data
        logging.info('Reading member info')
        with open(path.join(ds_dir, '%s.json' % Member().export_filename), 'r') as membs_f:
            for l in membs_f:
                d = convert(json.loads(l))
                m_id = d['id']
                del d['id']
                # ids in the seed list which could not be crawled are added here
                membs.add(m_id)
                if 'badges' in d:
                    badges = d['badges']
                    for b in badges:
                        # store the badge "THE" as THE
                        if 'THE' not in b:
                            d[b] = True
                        else:
                            d['THE'] = True
                    del d['badges']
                if 'helpfulStat' in d:
                    h = d['helpfulStat']
                    d['helpfulRatio'], d['helpfulTotal'] = float(h[0])/h[1], h[1]
                    del d['helpfulStat']
                review_graph.add_node(m_id, **d)
        # On some people profiles the number of reviews is not shown while they have reviews
        for m in membs:
            review_stat = max(d.get('reviewStat', 0), len(nx.neighbors(review_graph, m)) if m in review_graph else 0)
            review_graph.add_node(m, reviewStat=review_stat)
        logging.info('Reading product info')
        with open(path.join(ds_dir, '%s.json' % Product().export_filename), 'r') as prods_f:
            for l in prods_f:
                d = convert(json.loads(l))
                p_id = d['id']
                del d['id']
                # ids in the seed list which could not be crawled are added here
                prods.add(p_id)
                review_graph.add_node(p_id, **d)
        # making sure nReviews is not less than crawled reviews if it is scraped incorrectly
        for p in prods:
            n_reviews = max(d.get('nReviews', 0), len(nx.neighbors(review_graph, p)) if p in review_graph else 0)
            review_graph.add_node(p, nReviews=n_reviews)
    return review_graph, membs, prods


def crawl_to_graph_file(out_graph_file='review_graph', **kwargs):
    """
    Writes the crawl output into a graph file
    :param out_graph_file: name of the output graph file
    :param kwargs:
    """
    graph, membs, prods = crawl_to_graph(**kwargs)
    logging.info('Writing the graph to file')
    nx.write_gexf(graph, path.join(settings.DATA_SET_DIR, '%s.gexf' % out_graph_file), version='1.2draft', encoding='us-ascii')
    logging.info('Pickling the member and product id lists')
    with open(path.join(settings.DATA_SET_DIR, '%s.pickle' % MEMB_ID_F), 'w') as membs_ids_f:
        pickle.dump(membs, membs_ids_f)
    with open(path.join(settings.DATA_SET_DIR, 'product_ids.pickle'), 'w') as prods_ids_f:
        pickle.dump(prods, prods_ids_f)
    logging.info('Writing finished')


def read_graph_file(graph_file='review_graph'):
    logging.info('Reading from the graph file')
    review_graph = nx.read_gexf(path.join(settings.DATA_SET_DIR, '%s.gexf' % graph_file))
    logging.info('Unpickling the member and product id lists')
    with open(path.join(settings.DATA_SET_DIR, 'member_ids.pickle'), 'r') as membs_ids_f:
        membs = pickle.load(membs_ids_f)
    with open(path.join(settings.DATA_SET_DIR, 'product_ids.pickle'), 'r') as prods_ids_f:
        prods = pickle.load(prods_ids_f)
    return review_graph, membs, prods
