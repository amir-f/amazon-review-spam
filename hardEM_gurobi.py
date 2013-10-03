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
ORIG_LABEL = 'orig_label'


class FC:
    """
    Factor
    """
    def __init__(self, name):
        self.name = name
        self.rand_init()

    def rand_init(self):
        pass

    def __hash__(self):
        return self.name


class BinaryFC(FC):
    def __init__(self, name, author_graph):
        self.author_graph = author_graph
        FC.__init__(self, name)

    def rand_init(self):
        p = np.clip(dirichlet([1]*2)[0], EPS, 1 - EPS)
        self.log_val = np.log(p)
        self.log_1_val = np.log(1 - p)

    def log_likelihood(self, a):
        if self.author_graph.node[a][self.name]:
            return self.log_val
        else:
            return self.log_1_val

    def m_step(self, author_ids, em):
        if not author_ids:
            self.log_val = self.log_1_val = -1
            return
        author_graph = em.author_graph
        cnt = 0
        for a in author_ids:
            if author_graph.node[a][self.name]:
                cnt += 1
        v = np.clip(float(cnt) / len(author_ids), EPS, 1 - EPS)
        self.log_val = np.log(v)
        self.log_1_val = np.log(1 - v)


class NormFC(FC):
    def __init__(self, name, author_graph, mu_range_prior):
        self.author_graph = author_graph
        self.mu_range = mu_range_prior
        FC.__init__(self, name)

    def rand_init(self):
        self.mu = random.uniform(*self.mu_range)
        self.sigma2 = HardEM.CLUSTER_SIGMA

    def log_likelihood(self, a):
        return -((self.author_graph.node[a][self.name] - self.mu) ** 2) / (2 * self.sigma2 + EPS) - (log_2pi + log(self.sigma2)) / 2.0

    def m_step(self, author_ids, em):
        if not author_ids:
            self.rand_init()
            return
        self.mu, self.M2 = 0, 0
        freq = 0
        for a in author_ids:
            freq += 1
            author = em.author_graph.node[a]
            delta = author['revLen'] - self.mu
            self.mu += delta / freq
            self.M2 += delta * (author['revLen'] - self.mu)     # old_delta * new_delta
        self.sigma2 = HardEM.CLUSTER_SIGMA  # self.M2 / (len(author_ids) - 1 + EPS) + EPS


class ProdsFC(FC):
    def __init__(self, name, author_graph, author_product_map):
        assert set(author_product_map) == set(range(len(author_product_map)))
        self.author_product_map = author_product_map
        all_products = set.union(*[set(self.author_product_map[a]) for a in author_graph])
        self.n_all_products = len(all_products)
        FC.__init__(self, name)

    def rand_init(self):
        self.log_pr_prod = np.log(dirichlet([HardEM.PROD_PRIOR_ALPHA] * self.n_all_products))  # near uniform initialization

    def log_likelihood(self, a):
        if len(self.author_product_map[a]):
            return np.sum(self.log_pr_prod[self.author_product_map[a]])
        else: return 0

    def m_step(self, authors_ids, _):
        prod_freq = np.zeros(self.n_all_products)
        for a in authors_ids:
            if len(self.author_product_map[a]):
                increment = np.zeros(self.n_all_products)
                increment[self.author_product_map[a]] = 1
                prod_freq += increment
        prod_freq += EPS    # to avoid log of zero
        s = np.sum(prod_freq)
        self.log_pr_prod = np.log(prod_freq / s)


class MembsFC(FC):
    def __init__(self, name, author_graph):
        self.n_all_membs = len(author_graph)
        self.author_graph = author_graph
        FC.__init__(self, name)

    def rand_init(self):
        self.log_pr_prod = np.log(dirichlet([HardEM.PROD_PRIOR_ALPHA] * self.n_all_membs))  # near uniform initialization

    def log_likelihood(self, a):
        return np.sum(self.log_pr_prod[self.author_graph.neighbors(a)])

    def m_step(self, authors_ids, _):
        memb_freq = np.zeros(self.n_all_membs)
        for a in authors_ids:
            increment = np.zeros(self.n_all_membs)
            increment[self.author_graph.neighbors(a)] = 1
            memb_freq += increment
        memb_freq += EPS    # to avoid log of zero
        s = np.sum(memb_freq)
        self.log_pr_prod = np.log(memb_freq / s)


class ClusterPrior:
    def __init__(self, v):
        self.log_v = np.log(v)

    def log_likelihood(self, _):
        return self.log_v

    def m_step(self, partition_authors, em):
        v = np.clip(float(len(partition_authors)) / len(em.author_graph), EPS, 1)
        self.log_v = np.log(v)


class HardEM:
    # class parameters and their default values
    EM_RESTARTS = 16
    EM_ITERATION_LIMIT = 30
    LP_TIME_LIMIT = 60
    LP_ITERATION_LIMIT = 50 * (10 ** 3)
    LP_VERBOSITY = 2
    DFLT_NPARTS = 5
    DEFLT_TAU = 0.7
    CLUSTER_SIGMA = 0.6
    PROD_PRIOR_ALPHA = 10
    CLUSTER_PRIOR_ALPHA = 10
    DENOM_THRESH = 3

    def __init__(self, author_graph, author_product_map, nparts=DFLT_NPARTS, init_partition=None, TAU=DEFLT_TAU, parallel=False):
        self.parts = range(nparts)
        self.TAU = TAU
        self.author_graph = author_graph
        self.author_product_map = author_product_map
        # if run in parallel set the random seed to pids. Otherwise, all instances will have same seed based on time
        if parallel:
            random.seed(os.getpid())
            np.random.seed(os.getpid())
        # assert numerical values for node ids
        assert set(author_graph) == set(range(len(author_graph)))
        self._lp_inited = False
        # init hidden vars
        if init_partition:
            self.partition = np.array(init_partition, dtype=np.int8)
            self.rand_init_param()
            self.m_step()   # so thetas have value
        else:
            self.partition = np.zeros(len(self.author_graph), dtype=np.int8)
            self.rand_init_param()

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

    def rand_init_param(self):
        logging.debug('Random param with seed: %s' % os.getpid())
        self.factors = [list() for _ in self.parts]
        # init cluster prior
        for p, prob in enumerate(dirichlet([HardEM.CLUSTER_PRIOR_ALPHA] * len(self.parts))):
            self.factors[p].append(ClusterPrior(prob))
        # init other singleton potential factors
        for p in self.parts:
            factors = self.factors[p]
            # factors.append(Binary_FC('isRealName', self.author_graph))
            # factors.append(Norm_FC('revLen', self.author_graph, (3, 7)))
            factors.append(ProdsFC('prProds', self.author_graph, self.author_product_map))
            factors.append(MembsFC('prMembs', self.author_graph))

    def _init_LP(self):
        if self._lp_inited:
            return

        logging.debug('Init LP')
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
        s = np.log(1 - self.TAU) - np.log(self.TAU)
        lpcoeffs, lpvars = [], []
        for a, b in self.author_graph.edges():
            lpcoeffs.append(-self.author_graph[a][b]['weight'] * s)
            lpvars.append(beta2[(a, b)])
        self.objF_pair = LinExpr(list(lpcoeffs), list(lpvars))

        self._lp_inited = True
        logging.debug('Init LP Done')

    def log_phi(self, a, p):
        return sum(factor.log_likelihood(a) for factor in self.factors[p])

    def log_likelihood(self):
        ll = sum(self.log_phi(a, self.partition[a]) for a in self.author_graph)
        log_TAU, log_1_TAU = np.log(self.TAU), np.log(1 - self.TAU)
        for a, b in self.author_graph.edges():
            if self.partition[a] == self.partition[b]:
                ll += self.author_graph[a][b]['weight'] * log_TAU
            else:
                ll += self.author_graph[a][b]['weight'] * log_1_TAU
        #for a, b in self.author_graph.edges():
        #    if self.partition[a] != self.partition[b]:
        #        ll += self.author_graph[a][b]['weight'] * (log_1_TAU - log_TAU)
        return ll

    def e_step(self):
        logging.debug('E-Step')
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
        logging.debug('E-Step Done')

    def m_step(self):
        logging.debug('M-Step')
        # create lists of nodes per cluster/partition
        partition = [list() for _ in self.parts]
        for a in self.author_graph:
            partition[self.partition[a]].append(a)
        # run m-step on factors of each cluster
        for p in self.parts:
            for factor in self.factors[p]:
                factor.m_step(partition[p], self)
        logging.debug('M-Step Done')

    def iterate(self, MAX_ITER=20):
        past_ll, past_partition = -float('inf'), -1 * np.ones(self.partition.size)
        ll, partition = self.log_likelihood(), self.partition.copy()
        logging.info('init \tlog_l: %s' % ll)
        EPS_CHNG = 1e-3
        itr = 0
        while (float(sum(partition != past_partition)) / partition.size > EPS_CHNG or abs(ll - past_ll) > EPS_CHNG)\
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
    def _preprocess_graph_and_map(author_graph, author_product_map):
        # remove weak edges
        author_graph.remove_edges_from([e for e in author_graph.edges(data=True) if e[2]['denom'] < HardEM.DENOM_THRESH])
        author_graph = nx.convert_node_labels_to_integers(author_graph, label_attribute=ORIG_LABEL)
        for aa in author_graph:
            a = author_graph.node[aa][ORIG_LABEL]
            prods = author_product_map[a]
            del author_product_map[a]
            author_product_map[aa] = prods
        # relabel author_product_map values so the product ids start with zero and there is no gap in the range
        author_product_map = HardEM._relabel_to_int_product_ids(author_product_map)
        return author_graph, author_product_map

    @staticmethod
    def run_EM(author_graph, author_product_map, nparts=DFLT_NPARTS, em_restarts=EM_RESTARTS,
               em_max_iter=EM_ITERATION_LIMIT, TAU=DEFLT_TAU, init_partition=None, parallel=True, nprocs=None):
        # setup the values
        author_graph, author_product_map = HardEM._preprocess_graph_and_map(author_graph.copy(), author_product_map.copy())
        map_input = [{'constr': {'author_graph': author_graph, 'author_product_map': author_product_map, 'nparts': nparts,
                                 'init_partition': init_partition, 'TAU': TAU, 'parallel': parallel},
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
        node_to_partition = {author_graph.node[n][ORIG_LABEL]: partition[n] for n in author_graph}

        return ll, node_to_partition


def em_create_n_iterate(args):
    em = HardEM(**args['constr'])
    return em.iterate(MAX_ITER=args['itr'])


def ll_partition_max_ll(t1, t2):
    if t1[0] >= t2[0]:
        return t1
    else:
        return t2