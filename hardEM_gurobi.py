import networkx as nx
from gurobipy import Model as LPModel, GRB, LinExpr
from numpy import log, pi
from multiprocessing import Pool
import numpy as np
import random
import os
from numpy.random.mtrand import dirichlet
import logging

logging.basicConfig(level=logging.INFO, format='%(process)d\t%(asctime)s:%(levelname)s: %(message)s', datefmt='%H:%M:%S')

EPS = 1e-12
log_2pi = log(2 * pi)


class hard_EM:
    # class parameters and their default values
    EM_RESTARTS = 8
    EM_ITERATION_LIMIT = 30
    LP_TIME_LIMIT = 60
    LP_ITERATION_LIMIT = 50 * (10 ** 3)
    LP_VERBOSITY = 2
    DEF_NPARTS = 5
    DEF_TAU = 0.9
    CLUSTER_SIGMA = 0.6
    DIRICH_PARAM = 1000
    DENOM_THRES = 3

    def __init__(self, author_graph, author_product_map, TAU=DEF_TAU,  nparts=DEF_NPARTS, init_partition=None):
        self.parts = range(nparts)
        self.TAU = TAU
        # assert numerical values for node ids
        assert set(author_graph) == set(range(len(author_graph)))
        assert set(author_product_map) == set(range(len(author_product_map)))
        self.author_graph = author_graph
        self.author_product_map = author_product_map
        all_products = set.union(*[set(self.author_product_map[a]) for a in author_graph])
        self.n_all_products = len(all_products)
        self._lp_inited = False
        # init hidden vars
        if init_partition:
            self.partition = init_partition
            self.m_step()   # so thetas have value
        else:
            self.partition = np.zeros(len(self.author_graph), dtype=np.int8)
            self._rand_init_param()

    @staticmethod
    def _relabel_to_int_product_ids(mapping):
        new_map = {}
        label_map = {}
        for k, vs in mapping.items():
            new_vs = []
            for v in vs:
                if v not in label_map:
                    label_map[v] = len(label_map)
                new_vs.append(label_map[v])
            new_map[k] = np.array(new_vs)
        return new_map

    def _rand_init_param(self):
        logging.debug('Random param with seed: %s' % os.getpid())
        random.seed(os.getpid())
        self.theta = [dict() for p in self.parts]
        for p, prob in enumerate( np.log(dirichlet([10] * len(self.parts))) ):
            self.theta[p]['logPr'] = prob
        for p in self.parts:
            prH, prR, prV = np.clip([dirichlet([1]*2)[0], dirichlet([1]*2)[0], dirichlet([1]*2)[0]], EPS, 1 - EPS)
            self.theta[p]['logPrH'] = log(prH)
            self.theta[p]['log1-PrH'] = log(1 - prH)
            self.theta[p]['logPrR'] = log(prR)
            self.theta[p]['log1-PrR'] = log(1 - prR)
            self.theta[p]['logPrV'] = log(prV)
            self.theta[p]['log1-PrV'] = log(1 - prV)
            self.theta[p]['muL'] = random.uniform(0, 6)
            self.theta[p]['sigma2L'] = hard_EM.CLUSTER_SIGMA  # random.uniform(0, 2)
            self.theta[p]['PrProd'] = np.log(dirichlet([hard_EM.DIRICH_PARAM] * self.n_all_products))  # near uniform initialization

    def _init_LP(self):
        if self._lp_inited:
            return

        logging.info('Init LP')
        self.lp = LPModel('estep')
        self.lp.setAttr("modelSense", 1)    # minimzation

        self.alpha = {}
        beta2 = {}
        beta3 = {}
        # instantiate vars
        logging.debug('Init LP - create vars')
        for a in self.author_graph:
            self.alpha[a] = {}
            for p in self.parts:
                self.alpha[a][p] = self.lp.addVar(lb=0.0)
        for a, b in self.author_graph.edges():
            beta2[(a, b)] = self.lp.addVar()
            beta3[(a, b)] = {}
            for p in self.parts:
                beta3[(a, b)][p] = self.lp.addVar(lb=0.0)
        # integrate added variables into the model
        self.lp.update()
        # add constraints once during this init
        # alphas are indicator vars
        logging.debug('Init LP - indiv constraints')
        ones_arr = [1.0] * len(self.parts)
        for a in self.author_graph:
            self.lp.addConstr(LinExpr(ones_arr, self.alpha[a].values()), GRB.EQUAL, 1.0)
        # beta2 is the sum of beta3s
        logging.debug('Init LP - pair constraints')
        pt_five_array = [0.5] * len(self.parts)
        for a, b in self.author_graph.edges():
            self.lp.addConstr(LinExpr(pt_five_array, beta3[(a, b)].values()), GRB.EQUAL, beta2[(a, b)])
            for p in self.parts:
                self.lp.addConstr(LinExpr([1.0, -1.0], [self.alpha[a][p], self.alpha[b][p]]), GRB.LESS_EQUAL, beta3[(a, b)][p])
                self.lp.addConstr(LinExpr([-1.0, 1.0], [self.alpha[a][p], self.alpha[b][p]]), GRB.LESS_EQUAL, beta3[(a, b)][p])
        self.lp.update()
        # calculate pairwise potentials part of the objective
        # the optimization is to minimize negated log-likelihood = maximize the log-likelihood
        logging.debug('Obj func - pair potentials')
        s = log(1 - self.TAU) - log(self.TAU)
        lpcoeffs, lpvars = [], []
        for a, b in self.author_graph.edges():
            lpcoeffs.append(-self.author_graph[a][b]['weight'] * s)
            lpvars.append(beta2[(a, b)])
        self.objF_pair = LinExpr(list(lpcoeffs), list(lpvars))

        self._lp_inited = True
        logging.info('Init LP Done')

    def log_phi(self, a, p):
        author = self.author_graph.node[a]
        th = self.theta[p]
        res = th['logPr']
        if author['hlpful_fav_unfav']:
            res += th['logPrH']
        else:
            res += th['log1-PrH']
        if author['isRealName']:
            res += th['logPrR']
        else:
            res += th['log1-PrR']
        # if author['vrf_prchs_fav_unfav']:
        #     res += th['logPrV']
        # else:
        #     res += th['log1-PrV']
        res += -((author['revLen'] - th['muL']) ** 2) / (2 * th['sigma2L'] + EPS) - (log_2pi + log(th['sigma2L'])) / 2.0
        res += np.sum(th['PrProd'][self.author_product_map[a]])
        return res

    def log_likelihood(self):
        ll = sum(self.log_phi(a, self.partition[a]) for a in self.author_graph.nodes())
        log_TAU, log_1_TAU = log(self.TAU), log(1 - self.TAU)
        for a, b in self.author_graph.edges():
            if self.partition[a] == self.partition[b]:
                ll += self.author_graph[a][b]['weight'] * log_TAU
            else:
                ll += self.author_graph[a][b]['weight'] * log_1_TAU
        return ll

    def e_step(self):
        logging.info('E-Step')
        if not self._lp_inited:
            self._init_LP()

        logging.debug('Obj func - indiv potentials')
        # individual potentials
        lpcoeffs, lpvars = [], []
        for a in self.author_graph:
            for p in self.parts:
                lpcoeffs.append(-self.log_phi(a, p))
                lpvars.append(self.alpha[a][p])
        objF_indiv = LinExpr(lpcoeffs, lpvars)
        self.lp.setObjective(objF_indiv + self.objF_pair)

        # solve the LP
        logging.debug('Solving the LP')
        self.lp.optimize()
        logging.debug('Solving the LP Done')

        # hard partitions for nodes (authors)
        for a in self.author_graph:
            self.partition[a] = np.argmax([self.alpha[a][p].X for p in self.parts])
        logging.info('E-Step Done')

    def m_step(self):
        logging.info('M-Step')
        stat = {st: [0.0] * len(self.parts) for st in ['freq', 'hlpful', 'realNm', 'vrfprchs', 'muL', 'M2']}
        stat['prod_freq'] = [np.zeros(self.n_all_products) for p in self.parts]
        for a in self.author_graph:
            p = self.partition[a]
            author = self.author_graph.node[a]
            stat['freq'][p] += 1
            if author['hlpful_fav_unfav']: stat['hlpful'][p] += 1
            if author['vrf_prchs_fav_unfav']: stat['vrfprchs'][p] += 1
            if author['isRealName']: stat['realNm'][p] += 1
            delta = author['revLen'] - stat['muL'][p]
            stat['muL'][p] += delta / stat['freq'][p]
            stat['M2'][p] += delta * (author['revLen'] - stat['muL'][p])
            increment = np.zeros(self.n_all_products)
            increment[self.author_product_map[a]] = 1
            stat['prod_freq'][p] += increment

        self.theta = [dict() for p in self.parts]
        sum_freq = sum(stat['freq'][p] for p in self.parts)

        for p in self.parts:
            self.theta[p]['logPr'] = log(stat['freq'][p] / (sum_freq + EPS) + EPS)
            prH = stat['hlpful'][p] / (stat['freq'][p] + EPS)
            prR = stat['realNm'][p] / (stat['freq'][p] + EPS)
            prV = stat['vrfprchs'][p] / (stat['freq'][p] + EPS)
            prH, prR, prV = np.clip([prH, prR, prV], EPS, 1 - EPS)
            self.theta[p]['logPrH'] = log(prH)
            self.theta[p]['log1-PrH'] = log(1 - prH)
            self.theta[p]['logPrR'] = log(prR)
            self.theta[p]['log1-PrR'] = log(1 - prR)
            self.theta[p]['logPrV'] = log(prV)
            self.theta[p]['log1-PrV'] = log(1 - prV)
            self.theta[p]['muL'] = stat['muL'][p]
            self.theta[p]['sigma2L'] = hard_EM.CLUSTER_SIGMA  # stat['M2'][st] / (stat['freq'][st] - 1 + EPS) + EPS
            s = sum(stat['prod_freq'][p])
            if s > 0:
                stat['prod_freq'][p] += EPS    # to avoid log of zero
                self.theta[p]['PrProd'] = np.log(stat['prod_freq'][p] / sum(stat['prod_freq'][p]))
            else:   # no frequencies, so set the distribution to near uniform
                self.theta[p]['PrProd'] = np.log(dirichlet([hard_EM.DIRICH_PARAM] * self.n_all_products))

        logging.info('M-Step Done')

    def iterate(self, MAX_ITER=20):
        past_ll, past_partition = -float('inf'), -1 * np.ones(len(self.partition))
        ll, partition = self.log_likelihood(), self.partition.copy()
        EPS = 1e-3
        itr = 0
        while (float(sum(partition != past_partition)) / len(partition) > EPS or abs(ll - past_ll) > EPS)\
                and itr < MAX_ITER:
            if ll < past_ll:
                logging.warning('ll decreased')
            itr += 1
            self.e_step()
            self.m_step()
            past_ll, past_partition = ll, partition.copy()
            ll, partition = self.log_likelihood(), self.partition
            logging.info('itr #%s\tlog_l: %s\tdelta: %s' % (itr, ll, ll - past_ll))
        logging.info('iterations: %d' % itr)

        if itr == MAX_ITER:
            logging.info('Hit max iteration: %d' % MAX_ITER)

        return ll, self.partition

    @staticmethod
    def _prepare_graph_and_map(author_graph, author_product_map):
        # remove weak edges
        author_graph.remove_edges_from([e for e in author_graph.edges(data=True) if e[2]['denom'] < hard_EM.DENOM_THRES])
        author_graph = nx.convert_node_labels_to_integers(author_graph, discard_old_labels=False)
        for a, aa in author_graph.node_labels.items():
            prods = author_product_map[a]
            del author_product_map[a]
            author_product_map[aa] = prods
        # relabel author_product_map values so the product ids start with zero and there is no gap in the range
        author_product_map = hard_EM._relabel_to_int_product_ids(author_product_map)
        return author_graph, author_product_map

    @staticmethod
    def run_EM(author_graph, author_product_map, nparts=DEF_NPARTS, em_restarts=EM_RESTARTS,
               em_max_iter=EM_ITERATION_LIMIT, parallel=True, nprocs=None):
        # setup the values
        author_graph, author_product_map = hard_EM._prepare_graph_and_map(author_graph.copy(), author_product_map.copy())
        map_input = [{'constr': {'author_graph': author_graph, 'author_product_map': author_product_map, 'nparts': nparts},
                      'itr': em_max_iter}] * em_restarts
        if parallel:
            pool = Pool(processes=nprocs)
            ll_partitions = pool.map(em_create_n_iterate, map_input)
            ll, partition = reduce(ll_partition_max_ll, ll_partitions)
            pool.close()
            pool.join()
        else:
            ll_partitions = map(em_create_n_iterate, map_input)
            ll, partition = reduce(ll_partition_max_ll, ll_partitions)

        int_to_orig_node_label = {v: k for k, v in author_graph.node_labels.items()}
        node_to_partition = {int_to_orig_node_label[n]: partition[n] for n in author_graph}

        return ll, node_to_partition


def em_create_n_iterate(args):
    em = hard_EM(**args['constr'])
    return em.iterate(MAX_ITER=args['itr'])


def ll_partition_max_ll(t1, t2):
    if t1[0] >= t2[0]:
        return t1
    else:
        return t2