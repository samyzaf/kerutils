# networkx graph utilities
# used for generating and manipulating rectrees (rectilinear trees)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import rcParams
#from scipy.misc import imresize, imsave
import random
import itertools
import h5py
from ezprogbar import ProgressBar

rcParams['figure.figsize'] = 8,7
rcParams['axes.grid'] = True

def random_tree(n, fac=0.5):
    g = nx.grid_2d_graph(n, n, periodic=False)
    g.graph['shape'] = (n,n)
    k = int(fac * len(g.nodes()))
    for i in range(k):
        nodes2 = get_nodes_by_rank(g, 2)
        if nodes2:
            node = random.choice(get_nodes_by_rank(g, 2))
            g.remove_node(node)
            #draw_graph(g)
 
    #deprecated: comps = nx.connected_component_subgraphs(g, copy=True)
    comps = (g.subgraph(c).copy() for c in nx.connected_components(g))
    max_comp = None
    max_size = 0
    for c in comps:
        size = len(c.nodes())
        if size > max_size:
            max_comp = c
            max_size = size

    t = nx.minimum_spanning_tree(max_comp)
    return t

def graph_compaction(g):
    while True:
        compacted = False
        for ((x1,y1),(x2,y2)) in g.edges():
            ng = None
            if abs(x1-x2) > 1:
                if x1 > x2:
                    x1, x2 = x2, x1
                ng = shift_left(g, (x2,y2))
            elif abs(y1-y2) > 1:
                if y1 > y2:
                    y1, y2 = y2, y1
                ng = shift_down(g, (x2,y2))
            if ng is None:
                continue
            g = ng
            compacted = True
            break
        if not compacted:
            break

    rg = reset_axis(g)
    return rg

def get_nodes_by_rank(g, d):
    nodes = []
    for n in g.nodes():
        if g.degree(n) == d:
            nodes.append(n)
    return nodes

def get_corner_nodes(g):
    nodes = []
    for n in g.nodes():
        if g.degree(n) == 2:
            n1, n2 = g.neighbors(n)
            if distance(n1,n2) < 1.5:
                nodes.append(n)
    return nodes

def get_mid_nodes(g):
    nodes = []
    for n in g.nodes():
        if g.degree(n) == 2:
            n1, n2 = g.neighbors(n)
            if distance(n1,n2) > 1.9:
                nodes.append(n)
    return nodes

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

def elist2graph(elist):
    g = nx.Graph()
    for (x1,y1),(x2,y2) in elist:
        g.add_node((x1,y1))
        g.add_node((x2,y2))
        g.add_edge((x1,y1),(x2,y2))
    return g

def prune_graph(g):
    items = []
    for node in g.nodes():
        if not g.degree(node) == 2:
            continue
        (x1,y1), (x2,y2) = g.neighbors(node)
        if (not x1 == x2) and (not y1 == y2):
            continue
        items.append((node, (x1,y1), (x2,y2)))
    for node, v1, v2 in items:
        g.remove_node(node)
        g.add_edge(v1, v2)

def draw_graph(g, **opt):
    ax = opt.get('ax', None)
    savefile = opt.get('savefile', None)
    width = opt.get('width', 3)
    nodelist = opt.get('nodelist', 'auto')
    node_size = opt.get('node_size', 80)
    labels = opt.get('labels', 'none')

    pos = dict((n, n) for n in g.nodes())

    if nodelist == 'auto':
        nodelist = g.nodes()
    elif nodelist == 'deg1':
        nodelist = get_nodes_by_rank(g, 1)
        nodelist.extend(get_corner_nodes(g))
    elif nodelist == 'none':
        nodelist = []

    if labels == 'auto':
        labels = dict((n, "(%d,%d)" % n) for n in g.nodes())
    else:
        labels = None

    nx.draw(g, pos, font_size=18, font_weight='bold',
            ax=ax, node_size=node_size, node_shape='s',
            width=width, node_color='black', nodelist=nodelist)
    #if savefile is None:
    #    plt.show()
    #    plt.clf()
    #    plt.close()
    #else:
    #    plt.savefig(savefile)

def distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    d = np.hypot(x1-x2, y1-y2)
    #return int(round(d))
    return d

def data_iterator(h5file):
    f = h5py.File(h5file, 'r')
    num_images = f.get('num_images')[()]
    for i in range(num_images):
        img = np.array(f.get('img_%d' % i))
        elist = [(tuple(a), tuple(b)) for a,b in f.get('elist_'+str(i))[()]]
        g = elist2graph(elist)
        yield img, g
    f.close()

def rgb2gray(rgb):
    img = np.dot(rgb[..., 0:3], [0.299, 0.587, 0.114])
    img = img.astype(int)
    return img

def draw_img(img):
    plt.imshow(img, cmap='gray', interpolation='none')
    #plt.title("Class {}".format(y_test[0]))
    plt.axis('off')
    plt.show()

def draw_img2(img):
    plt.grid(True)
    plt.xticks(np.arange(0, 48))
    plt.yticks(np.arange(0, 48))
    plt.imshow(img, cmap='gray', interpolation='none')
    #plt.title("Class {}".format(y_test[0]))
    #plt.axis('off')
    #plt.set_xticks(numpy.arange(0, 1.0, 0.1))
    #plt.set_yticks(numpy.arange(0, 1.0, 0.1))
    plt.show()

#---- Topology stuff
# A topological form (or signature) of a rectilinear tree
# is the smallest rectree isomorphic to it.

def topology(g):
    prune_graph(g)
    X = sorted(set(x for (x,y) in g.nodes()))
    Y = sorted(set(y for (x,y) in g.nodes()))
    xmap = dict((x,i) for i,x in enumerate(X))
    ymap = dict((y,i) for i,y in enumerate(Y))

    t = nx.Graph()
    for ((x1,y1),(x2,y2)) in g.edges():
        n1 = (xmap[x1], ymap[y1])
        n2 = (xmap[x2], ymap[y2])
        t.add_edge(n1, n2)

    t = graph_compaction(t)
    return t

def shift_left(g, node):
    cap = left_neighbor(g, node)
    B = branch_nodes(g, node, cap)
    A = set(node for node in g.nodes() if not node in B)
    Bs = set((x-1,y) for (x,y) in B)
    if nclosure(g, A).intersection(nclosure(g, B, 1, 0)):
        return None
    ng = nx.Graph()
    for x,y in A.union(Bs):
        ng.add_node((x,y))
    for ((x1,y1),(x2,y2)) in g.edges():
        if (x1,y1) in B:
            x1 -= 1
        if (x2,y2) in B:
            x2 -= 1
        ng.add_edge((x1,y1),(x2,y2))

    prune_graph(ng)
    return ng

def shift_down(g, node):
    cap = bottom_neighbor(g, node)
    B = branch_nodes(g, node, cap)
    A = set(node for node in g.nodes() if not node in B)
    Bs = set((x,y-1) for (x,y) in B)
    if nclosure(g, A).intersection(nclosure(g, B, 0, 1)):
        return None
    ng = nx.Graph()
    for x,y in A.union(Bs):
        ng.add_node((x,y))
    for ((x1,y1),(x2,y2)) in g.edges():
        if (x1,y1) in B:
            y1 -= 1
        if (x2,y2) in B:
            y2 -= 1
        ng.add_edge((x1,y1),(x2,y2))

    prune_graph(ng)
    return ng

def nclosure(g, A, xoffset=0, yoffset=0):
    C = set()
    for ((x1,y1),(x2,y2)) in g.edges():
        if (x1,y1) in A and (x2,y2) in A:
            if x1 == x2:
                if y2 < y1:
                    y1, y2 = y2, y1
                for y in range(y1, y2+1):
                    C.add((x1,y))
            else:
                if x2 < x1:
                    x1, x2 = x2, x1
                for x in range(x1, x2+1):
                    C.add((x,y2))
    C = set((x - xoffset, y - yoffset) for (x,y) in C)
    return C

def reset_axis(g):
    if g.size() == 0:
        return g
    a = min(x for (x,y) in g.nodes())
    b = min(y for (x,y) in g.nodes())
    rg = nx.Graph()
    for (x,y) in g.nodes():
        rg.add_node((x-a, y-b))
    for ((x1,y1),(x2,y2)) in g.edges():
        rg.add_edge((x1-a,y1-b),(x2-a,y2-b))
    return rg

def branch_nodes(g, node, cap):
    B = set()
    B.add(node)
    visited = {cap, node}
    while True:
        C = set()
        for b in B:
            for n in g.neighbors(b):
                if n in visited:
                    continue
                visited.add(n)
                C.add(n)
        if not C:
            break
        B = B.union(C)
    return list(B)

def left_neighbor(g, node):
    x0, y0 = node
    for x,y in g.neighbors(node):
        if x < x0:
            return (x,y)

def bottom_neighbor(g, node):
    x0, y0 = node
    for x,y in g.neighbors(node):
        if y < y0:
            return (x,y)
