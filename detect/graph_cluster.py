import logging
import multiprocessing as mp
from multiprocessing import Pool
from numpy import *
import networkx as nx
import shared_ns as sh

__author__ = 'Amir'

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

def write_graph_to_file(graph, graph_file):
    """
    Write the graph to a GEXF file
    """
    # write the graph
    logging.info('Started writing the graph')
    logging.debug(r'graph_file: %s'%graph_file)
    nx.write_gexf(graph, graph_file)
    logging.info('Finished writing the graph')

def read_graph_from_file(graph_file):
    """
    Read graph from a GEXF file
    """
    # read the graph
    logging.info('Started reading the graph')
    logging.debug(r'graph_file: %s'%graph_file)
    graph = nx.read_gexf(graph_file)
    logging.info('Finished reading the graph')

    return graph


def graph_cluster_EM(graph, bipartite=False, use_rev_len=False, **kwarg):
    """
    Returns a graph where nodes have cluster attribute

    Arguments:
    N_CLUSTERS -- Number of clusters to find
    EM_ITER -- How long the EM iterations should continue
    EM_REP -- How many times should EM restart with random initialization?
    bipartite --
    """

    if bipartite: N_CLUSTERS_DFLT = (2,2)
    else: N_CLUSTERS_DFLT = 2
    N_CLUSTERS = kwarg.get('N_CLUSTERS', N_CLUSTERS_DFLT)
    EM_ITER = kwarg.get('EM_ITER', 10)
    EM_REP = kwarg.get('EM_REP', 6)

    graph_orig = graph.copy()
    # remove singleton nodes
    _gd = graph.degree()
    singletons = set([n for n in graph.nodes() if _gd[n] == 0])
    graph.remove_nodes_from(singletons)

    adj_mat_N = len(graph)

    if bipartite:
        part1 = filter(lambda n: graph.node[n]['part'] == 1, graph.nodes())
        part2 = filter(lambda n: graph.node[n]['part'] == 2, graph.nodes())
        # labels to integers which puts part1 before part2
        node_seq = part1 + part2
        nx.relabel_nodes( graph, dict(zip(node_seq, range(adj_mat_N))), copy=False )
    else:
        node_seq = graph.nodes()
        nx.convert_node_labels_to_integers(graph)

    # graph.adjacency list returns in order of graph.nodes() which are
    # not necessarily sorted, so we make the adj_list_r sorted by node ids
    adj_list_r, adj_list_c = zeros(adj_mat_N, dtype=object), zeros(adj_mat_N, dtype=object)
    for n, adj_l in zip(graph.nodes(), graph.adjacency_list()):
        adj_list_r[n] = adj_l
    if isinstance(graph, nx.DiGraph):
        for n, adj_l in zip(graph.nodes(), graph.reverse.adjacency_list()):
            adj_list_c[n] = adj_l
    else:
        for n, adj_l in zip(graph.nodes(), graph.adjacency_list()):
            adj_list_c[n] = adj_l

    # If review length are taken into account
    if use_rev_len:
        logging.info('Computing log(review length) lists')
        revlen = zeros(adj_mat_N, dtype=object)
        for i in range(adj_mat_N):
            rvl_n = map(lambda x: len(x[2]['reviewTxt']), graph.edges(i, data=True))
            rvl = log(array(rvl_n))
            revlen[i] = rvl
    else:
        revlen = None

    logging.info('Started cluster detection')
    if bipartite:
        ghat = _graph_cluster_EM_bipartite( adj_lists=(adj_list_r, adj_list_c), Cs=N_CLUSTERS, EM_ITER=EM_ITER,
                                            EM_REP=EM_REP, parts_sizes=(len(part1), len(part2)) , revlen=revlen)
    else:
        ghat = _graph_cluster_EM(adj_lists=(adj_list_r, adj_list_c), C=N_CLUSTERS, EM_ITER=EM_ITER, EM_REP=EM_REP,
                                 revlen=revlen)
    logging.info('Finished cluster detection')

    for i, n in enumerate(node_seq):
        graph_orig.node[n]['cluster'] = int(ghat[i])

    return graph_orig


def graph_cluster_evaluate( adj_lists, C, EM_ITER, EM_REP, ground_truth ):
    """
    Takes a graph and cluster parameters along with groundtruth data and
    returns the result of graph clustering and its accuracy

    Arguments:
    adj_mat -- Adjacency matrix of the graph which is to be clustered. It is a tuple
                of sparse csr and csc adjacency matrices
    C -- Number of clusters (should be predefined)
    EM_ITER -- How long the EM iterations should continue
    EM_REP -- How many times should EM restart with random initialization
    ground_truth -- a vector with the length of the number graph nodes. groundTruth(i)
                   is the actual cluster node i belongs to. 0 <= groundTruth(i) <= C-1

    Returns:
    ghat -- a vector of the size of the node where ghat[i] is the cluster
             number the i-th node is assigned to.
    accuracy -- Rand measure:
                (TP + TN)/(TP + FP + FN + TN)

    afayazi@tamu.edu
    """

    ghat = _graph_cluster_EM(adj_lists, C, EM_ITER, EM_REP)

    logging.info('Started evaluation')
    ghat = matrix( ghat ).T
    ghat += min(ghat) + 1                 # no cluster number is zero
    N = len(ghat)

    # calculating Rand Measure to evaluate clustering. We use as much matrix and
    # array operations as possible instead of loops for speed up

    # same_cluster[i,j] is 1 if i and j belong to the same cluster
    g_hat_t = (1./ghat).T
    same_cluster_hat = (ghat * g_hat_t) == 1
    ground_truth = matrix(ground_truth).T
    ground_truth += min(ground_truth) + 1   # no cluster number is zero
    ground_truth_t = (1./ground_truth).T
    same_cluster = (ground_truth * ground_truth_t) == 1

    p = (sum(same_cluster_hat) - N)/2       # main diagonal , symmetric matrix
    p_gt = (sum(same_cluster) - N)/2        # number of positive samples from ground truth
    tp = (sum(multiply(same_cluster, same_cluster_hat)) - N)/2
    fn = p_gt - tp
    tn = N*(N-1)/2 - p - fn

    rand_measure = float(tp + tn)/( N*(N-1)/2 )

    return ghat, rand_measure


def _initProcess(adj_out, adj_in, C, EM_ITER, revlen):
    sh.adj_out = adj_out
    sh.adj_in = adj_in
    sh.C = C
    sh.EM_ITER = EM_ITER
    sh.revlen = revlen


def _reduce_ghat_ll(t1, t2):
    if t1[1] >= t2[1]:
        return t1
    else:
        return t2


def _graph_cluster_EM(adj_lists, C, EM_ITER, EM_REP, revlen=None):
    """
    Performs graph clustering using the EM method and a mixture model multiple
    times with random initialization and returns the best result. The details
    are in the paper:
       "Mixture models and exploratory analysis in networks, M. E. J. Newman
       and E. A. Leicht, Proc. Natl. Acad. Sci. USA 104, 9564-9569 (2007)."

    Arguments:
    adj_lists -- Adjacency matrix of the graph which is to be clustered. Zero means
              there is no edge. Other values means an edge. Weights are not taken
              into account. It is a tuple of sparse csr and csc matrices
    C -- Number of clusters (should be predefined)
    EM_ITER -- How long the EM iterations should continue
    EM_REP -- How many times should EM restart with random initialization
    procs --

    Returns:
    g_hat -- a vector of the size of the node where g_hat[i] is the cluster
            number the i-th node is assigned to.

    afayazi@tamu.edu
    """

    logging.debug( r'C=%r, EM_ITER=%d, EM_REP=%d, revlen=%r' % (C, EM_ITER, EM_REP, revlen is not None) )
    # matrix is sparse, so we keep the adjacency list explicitly for performance
    adj_out, adj_in = adj_lists

    g_hat, ll = _process_pool_map_reduce(adj_out, adj_in, C, EM_ITER, EM_REP, revlen)

    return g_hat


def _graph_cluster_EM_bipartite(adj_lists, Cs, EM_ITER, EM_REP, parts_sizes, revlen=None):
    """
    Performs graph clustering using the EM method and a mixture model multiple
    times with random initialization and returns the best result. The details
    are in the paper:
       "Mixture models and exploratory analysis in networks, M. E. J. Newman
       and E. A. Leicht, Proc. Natl. Acad. Sci. USA 104, 9564-9569 (2007)."

    Arguments:
    adj_lists -- Adjacency matrix of the graph which is to be clustered. Zero means
              there is no edge. Other values means an edge. Weights are not taken
              into account. It is a tuple of sparse csr and csc matrices
    Cs -- Number of clusters (should be predefined)
    EM_ITER -- How long the EM iterations should continue
    EM_REP -- How many times should EM restart with random initialization
    procs --

    Returns:
    ghats -- a vector of the size of the node where ghat[i] is the cluster
            number the i-th node is assigned to.

    afayazi@tamu.edu
    """

    logging.debug( r'C=%r, EM_ITER=%d, EM_REP=%d, part_sizes=%r, revlen=%r' % (Cs, EM_ITER, EM_REP, parts_sizes, revlen is not None) )
    # matrix is sparse, so we keep the adjacency list explicitly for performance
    adj_mat_r, adj_mat_c = adj_lists
    adj_mat_N = adj_mat_r.shape[0]
    part1_size, part2_size = parts_sizes
    c1, c2 = Cs

    logging.info('Forming adjacency lists')
    adj_out_1 = adj_mat_r[range(part1_size)]
    adj_in_1 = adj_mat_c[range(part1_size)]
    for i in range(part1_size):
        adj_out_1[i] = array(adj_out_1[i]) - part1_size
        adj_in_1[i] = array(adj_in_1[i]) - part1_size
    adj_out_2 = adj_mat_r[range(part1_size, adj_mat_N)]
    adj_in_2 = adj_mat_c[range(part1_size, adj_mat_N)]
    if revlen is not None:
        revlen_1 = revlen[range(part1_size)]
        revlen_2 = revlen[range(part1_size, adj_mat_N)]
    else:
        revlen_1 = revlen_2 = None

    # base_c is there to push the cluster number of the second part up in order to distinguish them from the
    # first part cluster numbers
    ghat = zeros(adj_mat_N)
    for adj_out_i, adj_in_j, C, base_c, part, revlen_i in [(adj_out_1, adj_in_2, c1, 0, range(part1_size), revlen_1),
                                                 (adj_out_2, adj_in_1, c2, c1, range(part1_size,adj_mat_N), revlen_2)]:
        ghat_p, ll = _process_pool_map_reduce(adj_out_i, adj_in_j, C, EM_ITER, EM_REP, revlen_i)
        ghat[part] = ghat_p + base_c

    return ghat


def _process_pool_map_reduce(adj_out_i, adj_in_j, C, EM_ITER, EM_REP, revlen, procs=mp.cpu_count()):
    logging.info(r'Started a pool of %d workers'%procs)
    pool = Pool(processes=procs, initializer=_initProcess, initargs=(adj_out_i, adj_in_j, C, EM_ITER, revlen))
    all_ghat_lls = pool.map(_graph_cluster_EM_singlerun_dispatcher, range(EM_REP))
    ghat_p, ll = reduce(_reduce_ghat_ll, all_ghat_lls)
    logging.info(r'Best loglikelihood: %r'%ll)

    pool.terminate()
    return ghat_p, ll


def _graph_cluster_EM_singlerun_dispatcher(i):
    if sh.revlen is not None:
        return _graph_cluster_EM_revlen_singlerun(sh.adj_out, sh.adj_in, sh.C, sh.EM_ITER, sh.revlen, i)
    else:
        return _graph_cluster_EM_singlerun(sh.adj_out, sh.adj_in, sh.C, sh.EM_ITER, i)


def _graph_cluster_EM_singlerun(adj_out, adj_in, C, EM_ITER, pid=None):
    """
    Clusters a graph into a predefined number of clusters based edge patterns.
    
    Arguments:
    adj_mat -- Adjacency matrix of the graph which is to be clustered. Zero means
              there is no edge. Other values means an edge. Weights are not taken
              into account. This matrix better be sparse.
    C -- Predefined number of clusters
    EM_ITER -- Number of iterations before stopping the EM loop

    Returns:
    g -- a vector of the size of the node where g[i] is the cluster number for the
         i-th node is assigned to.
    loglikelihood -- The log likelihood of the full data (observation, latent variables)

    afayazi@tamu.edu
    """

    logging.debug('Entered %s'%_graph_cluster_EM_singlerun.__name__)
    IRange, JRange = len(adj_out), len(adj_in)
    q = zeros((IRange, C))

    # init
    if pid is not None:
        random.seed()
    pi_ = ones(C) / C
    DAMP = 20
    noise = random.rand(C)/DAMP
    pi_ += noise - mean(noise)
    theta = ones((C, JRange))/DAMP
    noise = random.rand(C,JRange)/DAMP
    noise -= tile(mean(noise, axis=1)[:,newaxis], (1, JRange))
    theta += noise
    theta /= tile(sum(theta, axis=1)[:,newaxis], (1, JRange))
    prev_loglikelihood = float('-inf')
    loglikelihood = None
    g = None

    for iter in range(EM_ITER):
        # Expectation
        for i in range (IRange):
            for r in range(C):
                q[i,r] = log(pi_[r]) + sum( log(theta[r,adj_out[i]]) )
            # shift up the exponents not to lose major ones to numeric underflow
            # (their max is negative)
            q[i,:] -= max(q[i,:])
            q[i,:] = exp(q[i,:])
            q[i,:] /= sum(q[i,:])

        # Maximization
        pi_ = mean(q, axis=0)
        for r in range(C):
            for j in range(JRange):
                theta[r,j] = sum(q[adj_in[j],r])
            theta[r,:] /= sum(theta[r,:])

        g = argmax(q, axis=1)

        # computing the log likelihood
        loglikelihood = sum( log(pi_[g]) )
        for i in range(IRange):
            loglikelihood += sum( log(theta[g[i], adj_out[i]]) )

        logging.info( r'delta LogL--%i--%d: %f'%(pid, iter, loglikelihood - prev_loglikelihood) )
        prev_loglikelihood = loglikelihood

    logging.info( r'Final LogL--%d --> %f'%(pid, loglikelihood) )

    return g, loglikelihood


def _graph_cluster_EM_revlen_singlerun(adj_out, adj_in, C, EM_ITER, revlen, pid=None):
    """
    Clusters using the EM method and takes review length into account

    revlen -- logarithm of the length of the review

    afayazi@tamu.edu
    """

    logging.debug('Entered %s'%_graph_cluster_EM_revlen_singlerun.__name__)
    IRange, JRange = len(adj_out), len(adj_in)
    q = zeros((IRange, C))

    # init
    if pid is not None:
        random.seed()
    pi_ = ones(C) / C
    DAMP = 50
    noise = random.rand(C)/DAMP
    pi_ += noise - mean(noise)
    theta = ones((C, JRange))/DAMP
    noise = random.rand(C,JRange)/DAMP
    noise -= tile(mean(noise, axis=1)[:,newaxis], (1, JRange))
    theta += noise
    theta /= tile(sum(theta, axis=1)[:,newaxis], (1, JRange))
    mu = zeros(C)
    sigma2 = ones(C)
    sum_revlen = array([sum(rv) for rv in revlen])
    out_degree = array([len(adj_out_i) for adj_out_i in adj_out])
    prev_loglikelihood = float('-inf')
    loglikelihood = None
    g = None

    for iter in range(EM_ITER):
        # Expectation
        for i in range (IRange):
            for r in range(C):
                q[i,r] = log(pi_[r]) + sum( log(theta[r,adj_out[i]]) )
                delta = revlen[i] - ones(len(revlen[i]))*mu[r]
                q[i,r] -= dot(delta, delta)/sigma2[r]
                # shift up the exponents not to lose major ones to numeric underflow
            # (their max is negative)
            q[i,:] -= max(q[i,:])
            q[i,:] = exp(q[i,:])
            q[i,:] /= sum(q[i,:])

        # Maximization
        pi_ = mean(q, axis=0)
        for r in range(C):
            for j in range(JRange):
                theta[r,j] = sum(q[adj_in[j],r])
            theta[r,:] /= sum(theta[r,:])
            mu[r] = dot(q[:,r], sum_revlen)/dot(q[:,r], out_degree)
            sigma2_num = 0
            for i in range(IRange):
                delta = revlen[i] - ones(len(revlen[i]))*mu[r]
                sigma2_num += q[i,r]*dot(delta, delta)
            sigma2_denom = dot(q[:,r], out_degree-1) or 1
            sigma2[r] = sigma2_num/sigma2_denom

        g = argmax(q, axis=1)

        # computing the log likelihood
        loglikelihood = sum( log(pi_[g]) )
        for i in range(IRange):
            loglikelihood += sum( log(theta[g[i], adj_out[i]]) )

        logging.info( r'delta LogL--%i--%d: %f'%(pid, iter, loglikelihood - prev_loglikelihood) )
        prev_loglikelihood = loglikelihood

    logging.info( r'Final LogL--%d --> %f'%(pid, loglikelihood) )

    return g, loglikelihood





__all__ = ['read_graph_from_file', 'write_graph_to_file', 'graph_cluster_EM', 'graph_cluster_evaluate']