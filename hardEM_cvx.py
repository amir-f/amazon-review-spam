from cvxopt.modeling import op as cvx_op, dot as cvx_dot, sum as cvx_sum
from cvxopt.modeling import variable as cvx_var, spmatrix as cvx_mat

import networkx as nx
import numpy as np
from numpy import log
from random import randint
from datetime import datetime
import logging


EPS = 0.001
log_2pi = log(2 * np.pi)


def slog(msg):
    print('%s: %s' % (datetime.now().strftime('%I:%M:%S %p'), msg))


class hard_EM:
    def __init__(self, author_graph, TAU=0.8, nparts=5, init_partition=None):
        self.parts = range(nparts)
        self.TAU = TAU
        self.author_graph = nx.convert_node_labels_to_integers(author_graph, discard_old_labels=False)
        self._lp_init = False
        # init hidden vars
        if init_partition:
            self.partition = init_partition
        else:
            self.partition = {}
            for a in self.author_graph:
                self.partition[a] = randint(0, nparts - 1)
        self._m_step()

    def _init_LP(self):
        logging.debug('Init LP')

        # add constraints once here
        # constraints
        self.constraints = []
        nprt = len(self.parts)
        self.alpha = cvx_var(size=len(self.author_graph) * nprt, name='alpha')
        self.beta2 = cvx_var(size=self.author_graph.size(), name='beta2')
        self.beta3 = cvx_var(size=self.author_graph.size() * nprt, name='beta3')
        # node ids are integers (handled in the constructor)
        logging.debug('Init LP - indiv constraints')
        # alphas are indicator vars
        for a in self.author_graph:
            self.constraints.append(cvx_sum(self.alpha[a*nprt + p] for p in self.parts) == 1)
        for a in self.alpha:
            self.constraints.append(a >= 0)

        # beta2 is the sum of beta3s
        logging.debug('Init LP - pair constraints')
        for ei, (a, b) in enumerate(self.author_graph.edges()):
        #            if self.author_graph[a][b]['denom'] <= 2:
        #                continue
            self.constraints.append(0.5 * cvx_sum(self.beta3[ei*nprt + p] for p in self.parts) == self.beta2[ei])
            for p in self.parts:
                self.constraints.append(self.alpha[a*nprt + p] - self.alpha[b*nprt + p] <= self.beta3[ei*nprt + p])
                self.constraints.append(self.alpha[b*nprt + p] - self.alpha[a*nprt + p] <= self.beta3[ei*nprt + p])

        # pre compute pair-wise costs
        logging.debug('Init LP - pair costs')
        self.c_pair = cvx_mat([0.0] * self.author_graph.size())
        s = log(1 - self.TAU) - log(self.TAU)
        for ei, (a, b) in enumerate(self.author_graph.edges()):
        #            if self.author_graph[a][b]['denom'] <= 2:
        #                continue
            self.c_pair[ei] = self.author_graph[a][b]['weight'] * s

        self._lp_init = True
        logging.debug('Init LP Done')

    def _log_phi(self, a, p):
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
        ll = sum(self._log_phi(a, self.partition[a]) for a in self.author_graph.nodes())
        log_TAU, log_1_TAU = log(self.TAU), log(1 - self.TAU)
        for a, b in self.author_graph.edges():
            if self.partition[a] == self.partition[b]:
                ll += log_TAU * self.author_graph[a][b]['weight']
            else:
                ll += log_1_TAU * self.author_graph[a][b]['weight']
        return ll

    def _e_step(self):
        logging.debug('E-Step')
        if not self._lp_init:
            self._init_LP()

        nprt = len(self.parts)
        # individual potentials
        logging.debug('Obj func - indiv potentials')
        c_indiv = cvx_mat([0.0] * len(self.author_graph) * nprt)
        for a in self.author_graph:
            c_indiv[a*nprt:(a+1)*nprt] = [self._log_phi(a, p) for p in self.parts]
        objF = -cvx_dot(c_indiv, self.alpha)
        # pairwise potentials
        logging.debug('Obj func - pair potentials')
        objF -= cvx_dot(self.c_pair, self.beta2)

        lp = cvx_op(objF, self.constraints)
        # if logging.getLogger('root').level != logging.DEBUG:
        #     lp.options['show_progress'] = False
        # solve the LP
        logging.debug('Solving the LP')
        lp.solve('sparse', 'glpk')
        logging.debug('Solving the LP Done')

        # hard partitions for nodes (authors)
        self.partition = {}
        for a in self.author_graph:
            self.partition[a] = np.argmax(self.alpha.value[a*nprt:(a+1)*nprt])
        logging.debug('E-Step Done')

    def _m_step(self):
        logging.debug('M-Step')
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

        logging.debug('M-Step Done')

    def iterate(self, MAX_ITER=20):
        past_ll = -float('inf')
        ll = self.log_likelihood()
        EPS = 0.1
        itr = 0
        while abs(ll - past_ll) > EPS and itr < MAX_ITER:
            if ll < past_ll:
                logging.warning('ll decreased')
            self._e_step()
            self._m_step()
            past_ll = ll
            ll = self.log_likelihood()
            logging.info('log_l: %s\tdelta: %s' % (ll, ll - past_ll))
            itr += 1

        if itr == MAX_ITER:
            logging.warn('Hit max iteration: %d' % MAX_ITER)