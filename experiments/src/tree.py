import networkx as nx
import numpy as np
from   itertools import chain
import tqdm
from   nltk.corpus import wordnet as wn
import copy
import networkx as nx
from   itertools import chain
from   plotly.offline import iplot
import plotly.graph_objs as go
import plotly
import igraph
from queue import Queue
from .utils import tosn, town

class WnTree(nx.DiGraph):
    def __init__(self, df):
        """
        """
        super(WnTree, self).__init__()
        self._init_nodes()
        self._init_edges()
        self._init_attr(df)
        self._df = df
    
    def copy(self):
        return copy.deepcopy(self)
    
    def _init_nodes(self):
        nodes = [town(x) for x in tqdm.tqdm(wn.all_synsets(pos="n"))]
        self.add_nodes_from(nodes)
        self.root = nodes[0]
        for node in self.nodes:
            self.node[node]["lemma"] = tosn(node).lemma_names()[0]
            self.node[node]["closures"] = {"all":{node}}

    def _init_edges(self):
        edges = []
        for syns in tqdm.tqdm(self.nodes):
            edges.extend([(syns, town(x)) for x in chain(tosn(syns).hyponyms(), 
                                                         tosn(syns).instance_hyponyms())])
        self.add_edges_from(edges)

    def _init_attr(self, df):
        for node in self.nodes:
            self.node[node]["closures"] = {
                "all"    : {node},
                "image"  : {node}  if node in df.index else set(),
                "test"   : {node}  if ((node in df.index)  and df.loc[node,'test']) else set(),
                "train"  : {node}  if ((node in df.index)  and df.loc[node,'train']) else set(),
                "sem"    : {node}  if ((node in df.index)  and df.loc[node,'sem']) else set()
            }
            if self.node[node]["closures"]["train"]:
                self.node[node]["type"]=3
            elif self.node[node]["closures"]["test"]:
                self.node[node]["type"]=2
            elif self.node[node]["closures"]["image"]:
                self.node[node]["type"]=1
            else:
                self.node[node]["type"]=0
        
        for node in tqdm.tqdm(list(nx.topological_sort(self))[::-1]):
            for child in self.successors(node):
                for k in self.node[node]["closures"]:
                    self.node[node]["closures"][k]|=self.node[child]["closures"][k]
            for k in self.node[node]["closures"]:
                self.node[node][k]=len(self.node[node]["closures"][k])

    def subtree(self, source=None, depth_limit=4):
        source = source if source else self.root
        nodes  = list(nx.dfs_preorder_nodes(self, source, depth_limit))
        subg   =  self.subgraph(nodes)
        subg.root = source
        subg._df = self._df
        return subg
        
    def mst(self):
        tree = self.copy()
        x = nx.minimum_spanning_arborescence(tree)
        tree.remove_edges_from(list(tree.edges))
        tree.add_edges_from(x.edges)
        tree.root = self.root
        tree._init_attr(self._df)
        return tree
        
    def prune_empty_branches(self, selecta=None):
        self = self.copy()
        selecta = selecta if selecta else lambda x: x["train"]+x["test"]==0
        bunch = [node for node in self.nodes if selecta(self.nodes[node])]
        self.remove_nodes_from(bunch)
        return self
        
    def prune_dummy_nodes(self, selecta=None):
        self = self.copy()
        selecta = selecta if selecta else lambda x: x["type"]<=1
        for node in list(self.nodes):
            successors = list(self.successors(node))
            predecessors = list(self.predecessors(node))
            #if (self.node[node]["type"]<=1) and \
            if selecta(self.node[node]) and \
              (len(successors)==1)      and \
              (len(predecessors)==1):
                self.remove_node(node)
                self.add_edge(predecessors[0], successors[0]) 
        return self
    
    def pairwise_dist(self):
        gen = nx.shortest_path_length(self.to_undirected())
        dist = np.zeros((len(self.nodes),)*2)
        dico = {k:i for i,k in enumerate(self.nodes)}
        for x,y in tqdm.tqdm(gen):
            for z in y:
                dist[dico[x],dico[z]]=y[z]
        return dist, dico

def plot(self, dotsize=10, inter=False, attr=["lemma", "train","test","image"]):
    nodes = list(nx.dfs_preorder_nodes(self, self.root))
    inodes = {k:i for i,k in enumerate(nodes)}
    edges  = [(inodes[x[0]],inodes[x[1]]) for x in list(self.edges)]
    ig = igraph.Graph(len(nodes), directed=True)
    ig.add_edges(edges)
    lay = ig.layout("tree", root=[0])
    colors = list(_infer_colors(self, nodes))
    labels = list(_infer_anno(self, nodes, attr))

    M  = max([i[1] for i in lay])
    Xn = [i[0] for i in lay]
    Yn = [2*M-i[1] for i in lay]
    Xe = list(chain(*((Xn[e.source], Xn[e.target], None) for e in ig.es)))
    Ye = list(chain(*[[Yn[e.source], Yn[e.target], None] for e in ig.es]))

    lines = go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='none'
                       )
    dots = go.Scatter(x=Xn,
                      y=Yn,
                      mode='markers',
                      name='',
                      marker=dict(symbol='circle-dot',
                                    size=dotsize, 
                                    color=colors,#'#6175c1',#list(self._infer_colors()),#'#6175c1',    #'#DB4551', 
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                      text=labels,
                      hoverinfo='text',
                      opacity=0.8
                      )
    
    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )
    
    layout = dict(title= 'Tree with Reingold-Tilford Layout',  
                  annotations=make_annotations(lay, labels),
                  font=dict(size=12),
                  showlegend=False,
                  xaxis=go.XAxis(axis),
                  yaxis=go.YAxis(axis),          
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)'          
                  )
    
    data = go.Data([lines, dots])
    fig  = dict(data=data, layout=layout)
    fig['layout'].update(annotations=make_annotations(lay, labels))
    if inter:
        iplot(fig)
    else:
        plotly.offline.plot(fig, filename="./tmp/plot_{}.html".format(self.node[self.root]["lemma"]))
        
def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = go.Annotations()
    M = max([pos[k][1] for k in range(len(pos))])
    for k in range(L):
        annotations.append(
            go.Annotation(
                text=text[k], # or replace labels with a different list for the text within the circle  
                x=pos[k][0], y=2*M-pos[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations 

def _infer_colors(self, nodes):
    for node in nodes:
        yield {0:"#000000",1:"#000000",2:"#005500", 3:"#550000"}[self.node[node]["type"]]

def _infer_anno(self, nodes, attr=["lemma", "train","test","image"]):
    for node in nodes:
        yield node + "<br>" + "<br>".join(["{}:{}".format(k,self.node[node][k]) for k in attr])

def get_leaves(self):
    return [x for x in self.nodes if len(list(self.successors(x)))==0]

def closure(self, node):
    return set(nx.dfs_preorder_nodes(self, node))

def pred_closure(self, node):
    return set(chain(*nx.all_simple_paths(self, self.root, node)))

def succ_closure(self, source):
    nodes = set(nx.dfs_preorder_nodes(self, source))
    todel = {source}
    for child in self.successors(source):
        _succ_closure(self, child, nodes, todel)
    return todel

def _succ_closure(self, node, nodes, todel):
    x = set(self.predecessors(node))
    if len(x-todel)==0: 
        todel.add(node)
        for child in self.successors(node):
            _succ_closure(self, child, nodes, todel)
    return todel

def del_branch(self, source, include=False):
    self = copy.deepcopy(self)
    nodes = succ_closure(self, source)
    if include:
        nodes.remove(source)
    self.remove_nodes_from(list(nodes))
    return self

def del_branches(tree, removes):
    for syns in removes:
        if removes[syns]=="include":
            tree = del_branch(tree, syns, include=True)
        else:
            tree = del_branch(tree, syns)
        print(len(tree.nodes), len(tree.edges), count_tpe(tree), syns)
    return tree

def select_split(self, selecta, covered=set()):
    """
        Selecta: function node -> bool
    """
    nodes =set()
    burned=covered.copy()
    for node in tqdm.tqdm(covered):
        burned|=closure(self, node)
        burned|=pred_closure(self, node)
    for node in tqdm.tqdm(list(nx.topological_sort(self))[::-1]):
        if (node not in burned) and selecta(self.node[node]):
            nodes.add(node)
            burned|=pred_closure(self, node)
            burned|=closure(self, node)
    return nodes

def test_split(self, tenodes, trnodes):
    # Assert no overlap between training and test set
    assert len(tenodes.intersection(trnodes))==0
    allnodes = trnodes.union(tenodes)
    # For allnodes, assert no child/parent in allnodes
    for node in allnodes:
        assert pred_closure(self, node).intersection(allnodes)=={node}
        assert closure(self, node).intersection(allnodes)=={node}
        
def count_tpe(tree):
    x=[0, 0, 0, 0]
    for node in tree.nodes:
        x[tree.node[node]["type"]]+=1
    return x

def select_subset(tree, superset, distribution, nitems, ratio=1):
    """
        Ratio : 1=same distribution. 0=inverse distribution
    """
    tree  = tree.mst()
    _init_superset(tree, superset)
    dispatch = {tree.root:nitems}
    subset=set()
    for node in tqdm.tqdm(nx.topological_sort(tree)):
        n = dispatch[node]
        children = list(tree.successors(node))
        if (n==0): # Skip sub-branches to which no class is assigned
            for child in children:
                dispatch[child]=0
        elif node in superset: # Add superset class to subset
            assert n==1 # Assess that the algo correctly assigned a unique class to the branch
            subset.add(node)
            for child in children:
                dispatch[child]=0
        elif (len(children)==0):
            pass
        else:
            capacity = np.asarray([tree.node[node]["tmp"] for node in children])
            distrib  = np.asarray([tree.node[node][distribution] for node in children])
            distrib  = get_distribution(distrib, ratio)
            #print("_______________")
            #print(n, capacity, distrib)
            disp = dispatch_func(capacity, distrib, n)
            #print(disp)
            for i,child in enumerate(children):
                dispatch[child]=disp[i]
    _clean_superset(tree)
    return subset

def dispatch_func(capacity, distrib, nitems):
    """
        Not pretty but it works well
    """
    distrib = np.floor(nitems*(distrib)).astype(int)
    distrib_= np.minimum(distrib, capacity)
    x = distrib_.sum()
    while x<nitems:
        diff = (distrib/distrib.sum())-(distrib_/distrib_.sum())
        caps = capacity-distrib_
        fail = True
        for i in np.argsort(diff):
            if caps[i] > 0:
                distrib_[i]+=1
                fail=False
                x = distrib_.sum()
                break
        assert not fail
    assert x==nitems 
    return distrib_

def _init_superset(self, superset):
    for node in self.nodes:
        self.node[node]["tmp"] = {node} if node in superset else set()
    for node in tqdm.tqdm(list(nx.topological_sort(self))[::-1]):
        for child in self.successors(node):
            self.node[node]["tmp"]|=self.node[child]["tmp"]
    for node in self.nodes:
        self.node[node]["tmp"]=len(self.node[node]["tmp"])
        
def _clean_superset(self):
    for node in self.nodes:
        del self.node[node]["tmp"]
        
def inv_distribution(x):
    y = (1-x)
    return np.nan_to_num(y/y.sum())

def get_distribution(x, ratio=1):
    x = np.nan_to_num(x/x.sum())
    return ratio*x + (1-ratio)*inv_distribution(x)

def get_tree(df, removes={}):
    tree = WnTree(df)
    print(len(tree.nodes), len(tree.edges), count_tpe(tree))
    tree = tree.prune_empty_branches()
    print(len(tree.nodes), len(tree.edges), count_tpe(tree))
    tree = del_branches(tree, removes)
    tree = tree.prune_dummy_nodes()
    print(len(tree.nodes), len(tree.edges), count_tpe(tree))
    return tree