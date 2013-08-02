from pydev import pydevd
from analyses.graph_analysis import read_scrapy_write_gephi

__author__ = 'Amir'


if __name__ == '__main__':
#    pydevd.settrace('192.168.11.223', port=5690, stdoutToServer=True, stderrToServer=True)
    read_scrapy_write_gephi(dataset='spam2', graph_file_out='spam2_graph')
#    g = graphFactory(type='gephi')
#    print len(g.nodes())
