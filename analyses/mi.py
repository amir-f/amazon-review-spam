from numpy import nonzero, histogram, histogram2d
import numpy


def entropy(counts):
    """Compute entropy."""
    ps = counts / float(numpy.sum(counts))  # coerce to float and normalize
    ps = ps[nonzero(ps)]            # toss out zeros
    H = -numpy.sum(ps * numpy.log2(ps))   # compute entropy

    return H


def mutual_info_hist(x, y, bins):
    """Compute mutual information."""
    x = (x - min(x)) / float(max(x) - min(x))
    y = (y - min(y)) / float(max(y) - min(y))
    counts_xy = histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])[0]
    counts_x = histogram(x, bins=bins, range=[0, 1])[0]
    counts_y = histogram(y, bins=bins, range=[0, 1])[0]

    H_xy = entropy(counts_xy)
    H_x = entropy(counts_x)
    H_y = entropy(counts_y)

    return H_x + H_y - H_xy