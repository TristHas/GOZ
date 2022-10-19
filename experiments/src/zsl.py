import pickle, json
from math import ceil

import h5py, tqdm, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchtext

from .utils import makeadj, get_split

vismean = pd.read_pickle(vismean_path)
df_ = pd.read_pickle(fcgcn_path)

def pp(x):
    canvas =  "|".join([" {"+ str(i)+":.2f} " for i,y in enumerate(x)])
    print(canvas.format(*x))

def get_visu(syns):
    feats = vismean.loc[list(syns)].values
    return torch.from_numpy(feats).cuda().float()
    
def get_sem(syns, df):
    x = df.loc[list(syns)].fillna(0).values
    x = torch.from_numpy(x).cuda().float()
    return F.normalize(x)

def get_data(synsets, df, maxsamp=100, ratio=1):
    """"""
    feats, lbls = [],[]
    with h5py.File(storefile, "r") as f:
        for i,wnid in tqdm.tqdm(enumerate(synsets)):
            nsamp = min(ceil(f[wnid].shape[0]*ratio), maxsamp)
            feat  = f[wnid][:nsamp]
            feats += [torch.from_numpy(feat)]
            lbls  += [i]*feat.shape[0]
    lbls = torch.LongTensor(lbls).cuda().unsqueeze(1)
    feats= torch.cat(feats).cuda()
    sems = get_sem(synsets, df)
    return feats, sems, lbls

def get_split_data(split, df):
    """"""
    assert split in ["train", "all", "2-hops", "3-hops"]
    feats = torch.load(test_feats_path+split)
    lbls  = torch.load(test_lbls_path+split).unsqueeze(1)
    synsets = json.load(open(test_sets_path, 'r'))[split]#get_split(split)
    sems = get_sem(synsets, df)
    return feats, sems, lbls

def build_eszsl_features(split="2-hops", generalized=False):
    test_sets = json.load(open(test_sets_path, 'r'))
    test_wnids  = test_sets[split]#get_split(split)
    train_wnids = test_sets['train']#get_split("train")

    df_sem = pd.read_pickle(word_vectors_path)
    test_feats, s_te, test_lbls = get_split_data(split, df_sem)
    if generalized:
        test_wnids = train_wnids + test_wnids
        train_feats = torch.load(train_feats_path)
        train_lbls  = torch.load(train_lbls_path).unsqueeze(1)
        test_feats = torch.cat((train_feats, test_feats))
        test_lbls  = torch.cat((train_lbls, test_lbls+1000))
        s_te = get_sem(test_wnids, df_sem)
        
    v_tr = get_visu(train_wnids)
    s_tr = get_sem(train_wnids, df_sem)
    adjmat = makeadj(test_wnids)
    return test_feats, test_lbls, v_tr, s_tr, s_te, adjmat

def build_adgpm_features(split="2-hops", generalized=False):
    test_sets = json.load(open(test_sets_path, 'r'))
    test_wnids  = test_sets[split]#test_wnids  = get_split(split)
    train_wnids = test_sets['train']#train_wnids = get_split("train")
    df_sem = pd.read_pickle(fcgcn_path)
    test_feats, s_te, test_lbls = get_split_data(split, df_sem)
    
    if generalized:
        test_wnids = train_wnids + test_wnids
        train_feats = torch.load(train_feats_path)
        train_lbls  = torch.load(train_lbls_path).unsqueeze(1)
        test_feats = torch.cat((train_feats, test_feats))
        test_lbls  = torch.cat((train_lbls, test_lbls+1000))
        s_te = get_sem(test_wnids, df_sem)
        
    adjmat = makeadj(test_wnids)
    return test_feats, test_lbls, s_te[:,:2048], adjmat

def get_W(v_tr, s_tr, g=100, l=100):
    xx = v_tr.t().mm(v_tr)
    ss = s_tr.t().mm(s_tr)
    xs = v_tr.t().mm(s_tr)
    xx_inv = (xx + g * torch.eye(xx.size(0), dtype=torch.float32, device="cuda:0")).inverse()
    ss_inv = (ss + l * torch.eye(ss.size(0), dtype=torch.float32, device="cuda:0")).inverse()
    return xx_inv.mm(xs).mm(ss_inv)

def topk(XWY, k=5, lbl=None):
    if lbl is None:
        lbl = torch.arange(XWY.size(0)).long().cuda().unsqueeze(1)
    return torch.cumsum((XWY.topk(k)[1]==lbl).float().mean(0)*100, 0).tolist()

def acc_per_class(res, lbls, k=1):
    res = (torch.topk(res, k)[1] == lbls).long().sum(1)>0
    res = pd.Series(res.cpu().numpy(), index=lbls.squeeze().cpu().numpy())
    return res.groupby(res.index).mean().values

def scores(v_te, s_te, w):
    return v_te.mm(w).mm(s_te.transpose(0,1))

def test_eszsl_class(trnodes, tenodes, df):
    v_tr = get_visu(trnodes)
    v_te = get_visu(tenodes)
    s_te = get_sem(tenodes, df)
    s_tr = get_sem(trnodes, df)
    best = [0]
    for g in [0,1,10,100, 1000]:
        for l in [0,1,10,100, 1000]:
            w = get_W(v_tr, s_tr, g, l)
            res = scores(v_te, s_te, w)
            res = topk(res)
            if res[0]>best[0]:
                best=res
    return best

def test_eszsl_sample(trnodes, tenodes, df):
    v_tr = get_visu(trnodes)
    s_tr = get_sem(trnodes, df)
    v_te, s_te, l_te = get_data(tenodes, df)
    best = [0]
    for g in [0,1,10,100, 1000]:
        for l in [0,1,10,100, 1000]:
            w = get_W(v_tr, s_tr, g, l)
            res = scores(v_te, s_te, w)
            res = topk(res, 5, l_te)
            if res[0]>best[0]:
                best=res
    return best

def test_adgpm_class(trnodes, tenodes, df=None):
    df = df if df is not None else df_
    v_te = get_visu(tenodes)
    s_te = get_sem(tenodes, df)[:,:2048]
    best = [0]
    for g in [0,1,10,100, 1000]:
        for l in [0,1,10,100, 1000]:
            res = v_te.mm(s_te.t())
            res = topk(res)
            if res[0]>best[0]:
                best=res
    return best

def test_adgpm_sample(trnodes, tenodes, df=None):
    df = df if df is not None else df_
    v_te, s_te, l_te = get_data(tenodes, df)
    s_te = s_te[:,:2048]
    best = [0]
    for g in [0,1,10,100, 1000]:
        for l in [0,1,10,100, 1000]:
            res = v_te.mm(s_te.t())
            res = topk(res, 5, l_te)
            if res[0]>best[0]:
                best=res
    return best

def count_struct(res, lbl, adjmat):
    return _post_count(_count(res, lbl, adjmat))

def _count(res, lbl, adjmat):
    x,y = res.topk(1)[1].cpu().numpy(), lbl.cpu().numpy()
    z   = adjmat[x,y]
    x,y = np.unique(z.toarray(), return_counts=True)
    return pd.Series(y, index=x)

def _post_count(counts):
    counts = counts[[0,1,3,5]].fillna(0).values
    return counts/counts.sum()*100

def chunked_count(XWY, adjmat):
    counts = pd.Series(0, index=[0,1,3,5])
    for res, lbl in XWY:
        adds = _count(res, lbl, adjmat)
        counts = counts.add(adds, fill_value=0)
    return _post_count(counts)

def chunked_eszsl_scores(v_te, lbl, s_te, w, chunk=10000):
    for i in range(0, v_te.size(0), chunk):
        yield scores(v_te[i:i+chunk].cuda(), s_te, w), lbl[i:i+chunk].cuda()

def chunked_adgpm_scores(v_te, lbl, s_te, chunk=10000):
    for i in range(0,v_te.size(0), chunk):
        yield v_te[i:i+chunk].cuda().mm(s_te.t()).cuda(), lbl[i:i+chunk].cuda()

def chunked_eszsl(split, generalized, g=100, l=1):
    test_feats, test_lbls, v_tr, s_tr, s_te, adjmat \
     = build_eszsl_features(split=split, generalized=generalized)
    if generalized:
        te_msk = np.where(test_lbls.cpu().numpy()>1000)[0]
        test_feats=test_feats[te_msk]
        test_lbls=test_lbls[te_msk]
    w = get_W(v_tr, s_tr, g, l)
    res = chunked_eszsl_scores(test_feats, test_lbls, s_te, w)
    return chunked_count(res, adjmat)

def chunked_adgpm(split, generalized, g=100, l=1):
    test_feats, test_lbls, s_te, adjmat \
     = build_adgpm_features(split=split, generalized=generalized)
    if generalized:
        te_msk = np.where(test_lbls.cpu().numpy()>1000)[0]
        test_feats=test_feats[te_msk]
        test_lbls=test_lbls[te_msk]
    res = chunked_adgpm_scores(test_feats, test_lbls, s_te)
    return chunked_count(res, adjmat)

def plot(res):
    labels = 'Model error', 'Model error', 'Parent', 'Child'
    colors = ['red', 'blue', 'yellowgreen', 'orange']
    
    plt.pie(res, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')
    plt.show()

