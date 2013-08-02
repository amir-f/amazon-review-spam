__author__ = 'Amir'

import json

from libpgm.graphskeleton import GraphSkeleton
from libpgm.pgmlearner import PGMLearner

with open('data.txt', 'r') as f:
    data = eval(f.read())

# generate some data to use
skel = GraphSkeleton()
skel.load("skel.txt")
skel.toporder()

# instantiate my learner
learner = PGMLearner()

# estimate parameters from data and skeleton
result = learner.lg_mle_estimateparams(skel, data)

# output
print json.dumps(result.Vdata, indent=2)
