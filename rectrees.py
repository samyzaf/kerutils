import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import h5py

def data_iterator(h5file):
    f = h5py.File(h5file, 'r')
    num_images = f.get('num_images').value
    for i in range(num_images):
        img = np.array(f.get('img_%d' % i))
        elist = [(tuple(a), tuple(b)) for a,b in f.get('elist_'+str(i)).value]
        g = elist2graph(elist)
        yield img, g
    f.close()

# This is how we get a graph from an edgelist
def elist2graph(elist):
    g = nx.Graph()
    for (x1,y1),(x2,y2) in elist:
        g.add_node((x1,y1))
        g.add_node((x2,y2))
        g.add_edge((x1,y1),(x2,y2))
    return g

def get_nodes_by_deg(g, d):
    nodes = []
    for n in g.nodes():
        if g.degree(n) == d:
            nodes.append(n)
    return nodes

# This is how we get a sorted edgelist from a graph
def edgelist(g):
    edges = []
    for ((x1,y1),(x2,y2)) in g.edges():
        if x1 == x2:
            if y1 > y2:
                y1, y2 = y2, y1
        else:
            if x1 > x2:
                x1, x2 = x2, x1
        edges.append(((x1,y1), (x2,y2)))
    return sorted(edges)

def get_corner_nodes(g):
    nodes = []
    for n in g.nodes():
        if g.degree(n) == 2:
            n1, n2 = g.neighbors(n)
            if distance(n1,n2) < 1.5:
                nodes.append(n)
    return nodes

def draw_graph(g, **opt):
    ax = opt.get('ax', None)
    width = opt.get('width', 3)
    nodelist = opt.get('nodelist', 'auto')
    node_size = opt.get('node_size', 80)
    labels = opt.get('labels', 'none')

    pos = dict((n, n) for n in g.nodes())

    if nodelist == 'auto':
        nodelist = g.nodes()
    elif nodelist is 'deg1':
        nodelist = get_nodes_by_deg(g, 1)
        nodelist.extend(get_corner_nodes(g))
    elif nodelist is 'none':
        nodelist = []

    if labels == 'auto':
        labels = dict((n, "(%d,%d)" % n) for n in g.nodes())
    else:
        labels = None

    nx.draw(g, pos, font_size=18, font_weight='bold',
            ax=ax, node_size=node_size, node_shape='s',
            width=width, node_color='black', nodelist=nodelist)

def distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    d = np.hypot(x1-x2, y1-y2)
    return d
  
