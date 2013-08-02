import networkx as nx
from datetime import datetime
from sage import *
from sage.numerical.mip import MixedIntegerLinearProgram
import sage.numerical.backends.glpk_backend as backend
from sage.numerical.backends.generic_backend import get_solver
from numpy import log, pi
from random import randint
import multiprocessing as mp
from multiprocessing import Pool
import random
import os

EPS = 0.001
EM_RESTARTS = 8
EM_ITERATION_LIMIT = 5
LP_TIME_LIMIT = 60
LP_ITERATION_LIMIT = 5 * (10 ** 3)
log_2pi = log(2 * pi)


def slog(msg):
    print('%s\t%s: %s' % (os.getpid(), datetime.now().strftime('%I:%M:%S %p'), msg))


class hard_EM:
    def __init__(self, author_graph, TAU=0.5001, nparts=5, init_partition=None):
        self.parts = range(nparts)
        self.TAU = TAU
        self.author_graph = nx.convert_node_labels_to_integers(author_graph, discard_old_labels=False)
        self._lp_init = False
        # init hidden vars
        if init_partition:
            self.partition = init_partition
        else:
            self._rand_init_partition()
        self.m_step()

    def _rand_init_partition(self):
        slog('Random partitioning with seed: %s' % os.getpid())
        random.seed(os.getpid())
        self.partition = {}
        nparts = len(self.parts)
        for a in self.author_graph:
            self.partition[a] = randint(0, nparts - 1)

    def _init_LP(self):
        if self._lp_init:
            return

        slog('Init LP')
        self.lp = MixedIntegerLinearProgram(solver='GLPK', maximization=False)
        #self.lp.solver_parameter(backend.glp_simplex_or_intopt, backend.glp_simplex_only)       # LP relaxation
        self.lp.solver_parameter("iteration_limit", LP_ITERATION_LIMIT)
        # self.lp.solver_parameter("timelimit", LP_TIME_LIMIT)

    # add constraints once here
        # constraints
        self.alpha = self.lp.new_variable(dim=2)
        beta2 = self.lp.new_variable(dim=2)
        beta3 = self.lp.new_variable(dim=3)
        # alphas are indicator vars
        for a in self.author_graph:
            self.lp.add_constraint(sum(self.alpha[a][p] for p in self.parts) == 1)

        # beta2 is the sum of beta3s
        slog('Init LP - pair constraints')
        for a, b in self.author_graph.edges():
            if self.author_graph[a][b]['denom'] <= 2:
                continue
            self.lp.add_constraint(0.5 * sum(beta3[a][b][p] for p in self.parts) - beta2[a][b], min=0, max=0)
            for p in self.parts:
                self.lp.add_constraint(self.alpha[a][p] - self.alpha[b][p] - beta3[a][b][p], max=0)
                self.lp.add_constraint(self.alpha[b][p] - self.alpha[a][p] - beta3[a][b][p], max=0)

        # store indiv potential linear function as a dict to improve performance
        self.objF_indiv_dict = {}
        self.alpha_dict = {}
        for a in self.author_graph:
            self.alpha_dict[a] = {}
            for p in self.parts:
                var_id = self.alpha_dict[a][p] = self.alpha[a][p].dict().keys()[0]
                self.objF_indiv_dict[var_id] = 0        # init variables coeffs to zero

        # pairwise potentials
        slog('Obj func - pair potentials')
        objF_pair_dict = {}
        s = log(1 - self.TAU) - log(self.TAU)
        for a, b in self.author_graph.edges():
            if self.author_graph[a][b]['denom'] <= 2:
                continue
            var_id = beta2[a][b].dict().keys()[0]
            objF_pair_dict[var_id] = -self.author_graph[a][b]['weight'] * s
        self.objF_pair = self.lp(objF_pair_dict)

        self._lp_init = True
        slog('Init LP Done')

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
        res += -((author['revLen'] - th['muL']) ** 2) / (2 * th['sigma2L'] + EPS) - (log_2pi + log(th['sigma2L'])) / 2.0
        return res

    def log_likelihood(self):
        ll = sum(self.log_phi(a, self.partition[a]) for a in self.author_graph.nodes())
        log_TAU, log_1_TAU = log(self.TAU), log(1 - self.TAU)
        for a, b in self.author_graph.edges():
            if self.partition[a] == self.partition[b]:
                ll += log_TAU * self.author_graph[a][b]['weight']
            else:
                ll += log_1_TAU * self.author_graph[a][b]['weight']
        return ll

    def e_step(self):
        slog('E-Step')
        if not self._lp_init:
            self._init_LP()

        slog('Obj func - indiv potentials')
        # individual potentials
        for a in self.author_graph:
            for p in self.parts:
                self.objF_indiv_dict[self.alpha_dict[a][p]] = -self.log_phi(a, p)

        objF_indiv = self.lp(self.objF_indiv_dict)
        self.lp.set_objective(self.lp.sum([objF_indiv, self.objF_pair]))

        # solve the LP
        slog('Solving the LP')
        self.lp.solve(log=3)
        slog('Solving the LP Done')

        # hard partitions for nodes (authors)
        self.partition = {}
        for a in self.author_graph:
            membship = self.lp.get_values(self.alpha[a])
            self.partition[a] = max(membship, key=membship.get)
        slog('E-Step Done')

    def m_step(self):
        slog('M-Step')
        stat = {p: [0.0] * len(self.parts) for p in ['freq', 'hlpful', 'realNm', 'muL', 'M2']}
        for a in self.author_graph:
            p = self.partition[a]
            author = self.author_graph.node[a]
            stat['freq'][p] += 1
            if author['hlpful_fav_unfav']: stat['hlpful'][p] += 1
            if author['isRealName']: stat['realNm'][p] += 1
            delta = author['revLen'] - stat['muL'][p]
            stat['muL'][p] += delta / stat['freq'][p]
            stat['M2'][p] += delta * (author['revLen'] - stat['muL'][p])

        self.theta = [{p: 0.0 for p in ['logPr', 'logPrH', 'log1-PrH', 'logPrR', 'log1-PrR', 'sigma2L', 'muL']}
                      for p in self.parts]
        sum_freq = sum(stat['freq'][p] for p in self.parts)

        for p in self.parts:
            self.theta[p]['logPr'] = log(stat['freq'][p] / (sum_freq + EPS) + EPS)
            self.theta[p]['logPrH'] = log(stat['hlpful'][p] / (stat['freq'][p] + EPS) + EPS)
            self.theta[p]['log1-PrH'] = log(1 - stat['hlpful'][p] / (stat['freq'][p] + EPS) + EPS)
            self.theta[p]['logPrR'] = log(stat['realNm'][p] / (stat['freq'][p] + EPS) + EPS)
            self.theta[p]['log1-PrR'] = log(1 - stat['realNm'][p] / (stat['freq'][p] + EPS) + EPS)
            self.theta[p]['muL'] = stat['muL'][p]
            self.theta[p]['sigma2L'] = stat['M2'][p] / (stat['freq'][p] - 1 + EPS) + EPS

        slog('M-Step Done')

    def iterate(self, MAX_ITER=20):
        past_ll = -float('inf')
        ll = self.log_likelihood()
        EPS = 0.1
        itr = 0
        while abs(ll - past_ll) > EPS and itr < MAX_ITER:
            if ll < past_ll:
                slog('ll decreased')
            itr += 1
            self.e_step()
            self.m_step()
            past_ll = ll
            ll = self.log_likelihood()
            slog('itr #%s\tlog_l: %s\tdelta: %s' % (itr, ll, ll - past_ll))

        if itr == MAX_ITER:
            slog('Hit max iteration: %d' % MAX_ITER)

        return ll, self.partition

    def run_EM_pool(self, nprocs=mp.cpu_count()):
        pool = Pool(processes=nprocs)
        ll_partitions = pool.map(em_parallel_mapper, [self] * EM_RESTARTS)
        ll, partition = reduce(ll_partition_reducer, ll_partitions)
        pool.terminate()

        int_to_orig_node_label = {v: k for k, v in self.author_graph.node_labels.items()}
        node_to_partition = {int_to_orig_node_label[n]: partition[n] for n in partition}

        return ll, node_to_partition


def em_parallel_mapper(em_instance):
    if not em_instance._lp_init:
        em_instance._init_LP()
    em_instance._rand_init_partition()
    return em_instance.iterate(MAX_ITER=EM_ITERATION_LIMIT)


def ll_partition_reducer(t1, t2):
    if t1[0] >= t2[0]:
        return t1
    else:
        return t2