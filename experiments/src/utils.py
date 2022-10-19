import pickle, json
from itertools import chain
import xml.etree.ElementTree as ET

import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from nltk.corpus import wordnet as wn

from .path import *

parents_ = lambda x: {town(y) for y in x.closure(lambda x: chain(x.hypernyms(), x.instance_hypernyms()))}
children_= lambda x: {town(y) for y in x.closure(lambda x: chain(x.hyponyms(), x.instance_hyponyms()))}
parents  = lambda x: parents_(tosn(x))
children = lambda x: children_(tosn(x))
family   = lambda x: set.union(parents(x), children(x))
town = lambda x: x.pos() + str(x.offset()).zfill(8)
tosn = lambda x: wn.synset_from_pos_and_offset(pos=x[0], offset=int(x[1:]))

def get_imcount():
    et = ET.parse(ReleaseStatus)
    wnids, nIms = [],[]
    for i in et.iter():
        if i.tag=="synset":
            wnid = i.items()[0]
            ims  = i.items()[-1]
            assert ims[0]=='numImages'
            assert wnid[0]=='wnid'
            nIms.append(int(ims[1]))
            wnids.append(wnid[1])
    s = pd.Series(nIms, index=wnids)
    s = s.groupby(s.index).first()
    return s

def get_wrdcount(path=defcount):
    return pickle.load(open(path, "rb"))

def get_split(split):
    assert split in ["train", "all", "2-hops", "3-hops"]
    test_sets = json.load(open(sets_path, 'r'))
    wnids = test_sets[split]
    if split == "train":
        wnids.remove('n04399382')
    return wnids

def makeadj(test_wnids):
    test_set = set(test_wnids)
    par, chld = {}, {}
    maps = {k:i for i,k in enumerate(test_wnids)}
    for x in tqdm.tqdm(test_set):
        par[x] = list(set.intersection(parents(x), test_set))
        chld[x] = list(set.intersection(children(x), test_set))

    edges = []
    for x in tqdm.tqdm(test_set):
        edges.append((maps[x],maps[x], 1))
        
    for x,v in tqdm.tqdm(par.items()):
        for y in v:
            edges.append((maps[x],maps[y], 3))
            
    for x,v in tqdm.tqdm(chld.items()):
        for y in v:
            edges.append((maps[x],maps[y], 5))

    edges = np.asarray(edges).astype(int)

    return csc_matrix((edges[:,2], (edges[:,0], edges[:,1])))