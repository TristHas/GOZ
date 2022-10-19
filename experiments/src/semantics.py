import tqdm
import pickle, json
import numpy  as np
import pandas as pd
import torch, torchtext
from itertools import chain
from nltk.corpus import wordnet as wn
from .utils import tosn, get_imcount, get_wrdcount, get_split

def lemma_synsets(lem):
    """"""
    return [s.synset() for s in wn.lemmas(lem.name().lower())]

def synset_lemmas(syns):
    """"""
    return filter(lambda x: should_keep(x, syns, {}, {}), syns.lemmas())
 
def neigboring_synsets(syns):
    """"""
    return chain(syns.hypernyms(), syns.hyponyms())

def neighboring_lemmas(syns):
    """"""
    return chain(*[synset_lemmas(syns) for syns in neigboring_synsets(syns)])

def to_name(lems):
    """"""
    for lem in lems:
        yield lem.name().lower()

def avg_mean(lem, syns, glove):
    """"""
    tmp = glove.catch
    glove.catch=False
    names = list(to_name(neighboring_lemmas(syns)))
    x = glove[names]
    x /=torch.norm(x)
    y = glove[lem.name().lower()]
    y /=torch.norm(y)
    glove.catch=tmp
    return x.dot(y).item()

def lem_we_ok(lem, syns, glove):
    """"""
    syns = tosn(syns)
    others = lemma_synsets(lem)
    score  = avg_mean(lem, syns, glove)
    scores = [avg_mean(lem, syns, glove) for syns in others]
    return score==max(scores)

def is_first_meaning(lem, syns):
    """"""
    return wn.lemmas(lem.name())[0] is lem

def is_first_meaning_strict(lem, syns):
    """"""
    maxcount = [i.count() for i in wn.lemmas(lem.name().lower())]
    return lem.count() == max(maxcount)

def is_first_meaning_very_strict(lem, syns):
    """"""
    maxcount = [i.count() for i in wn.lemmas(lem.name().lower())]
    return (lem.count()==max(maxcount)) and lem.count() > 0

def is_not_first_meaning(lem, syns):
    """"""
    return not (wn.lemmas(lem.name())[0] is lem)

def is_not_first_meaning_strict(lem, syns):
    """"""
    maxcount = [i.count() for i in wn.lemmas(lem.name().lower())]
    return lem.count() != max(maxcount)

lem_wn_funcs = {"base":is_first_meaning,
                "strict":is_first_meaning_strict,
                "very":is_first_meaning_very_strict,
                "not":is_not_first_meaning,
                "not_strict":is_not_first_meaning_strict,
                "any": lambda x,y:True}

def word_min(word, counter, m):
    return counter[word]>m

def word_max(word, counter, m):
    return counter[word]<m

def word_uni(word):
    return not ("_" in word)

def word_ok(word, cond, counter=None):
    """
    """
    res = True
    if "max" in cond:
        res &= word_max(word, counter, cond["max"])
    if "min" in cond:
        res &= word_min(word, counter, cond["min"])
    if "voc" in cond:
        res &= cond["voc"](word)
    if "uni" in cond:
        res &= word_uni(word)
    return res

def lemma_ok(lem, syns, cond, glove=None):
    """
    """
    res = True
    if "wn" in cond:
        func = lem_wn_funcs[cond["wn"]]
        res &= func(lem, syns)
    if "we" in cond:
        if cond["we"]:
            res &= lem_we_ok(lem, syns, glove)
        else:
            res &= not(lem_we_ok(lem, syns, glove))
    return res

def should_keep(lem, syns, lem_cond, word_cond, counter=None, glove=None):
    """ """
    word  = word_ok(lem.name().lower(), word_cond, counter)
    lemma = lemma_ok(lem, syns, lem_cond, glove)
    return word and lemma

def select_lemmas(synsets, lem_cond, word_cond, counter=None, glove=None):
    """
        Outputs synswords
    """
    secondary = 0
    lost_syns = 0
    syns_words={}
    for syns in tqdm.tqdm(synsets):
        syns_words[syns]=[]
        for x in tosn(syns).lemmas():
            if should_keep(x, syns, lem_cond, word_cond, counter, glove):
                syns_words[syns].append(x.name().lower())            
            else:
                secondary+=1
        if len(syns_words[syns])==0:
            lost_syns+=1
    allwords = [x for k in syns_words for x in syns_words[k]]
    print("{} correct lemmas. ({} lemmas removed). {} synsets removed for lack of correct lemmas".format(len(allwords), secondary, lost_syns))
    syns_words=pd.Series([tuple(syns_words[k]) for k in syns_words], index=list(syns_words.keys()))
    return syns_words

def get_glove(synswords, glove):
    idx, vec = [],[]
    for wnid, words in synswords.iteritems():
        if len(words) > 0:
            idx.append(wnid)
            vec.append(glove[words].unsqueeze(0))
    vec = torch.cat(vec).numpy()
    return pd.DataFrame(vec, index=idx)

def base_glove(wnids, glove):
    synswords = {w: tosn(w).lemma_names() for w in wnids}
    synswords = pd.Series([tuple(synswords[k]) for k in synswords], 
                          index=list(synswords.keys()))
    tmp = glove.catch
    glove.catch=False
    df  = get_glove(synswords, glove)
    glove.catch = tmp
    return df

class GloVe():
    def __init__(self, cache="/home/tristan/data/word_embeddings/glove_vec/", 
                 name="6B", lower=True, catch=True):
        self.glove=torchtext.vocab.GloVe(cache=cache, name=name)
        self.catch=catch
        if lower:
            self.glove.stoi={k.lower():v for k,v in self.glove.stoi.items()}

    def isin(self, word):
        return word in self.glove.stoi
    
    def zeros(self):
        return torch.zeros(self.glove.dim)
    
    def __getitem__(self, words):
        if type(words) is str:
            words = [words]
        ret = self.zeros()
        cnt = 0
        for word in words:
            if self.isin(word):
                v = self.glove[word]
                ret += v
                cnt += 1
            else:
                if self.catch:
                    raise Exception
                else:
                    v = self._fix_word(word)
                    if v is not None:
                        ret += v
                        cnt += 1
        if cnt > 0:
            return ret / cnt
        else:
            if self.catch:
                raise Exception
            else:
                return ret
            
    def _fix_word(self, word):
        terms = word.replace('_', ' ').split(' ')
        ret = self.zeros()
        cnt = 0
        for term in terms:
            #v = self.embedding.get(term)
            v = None
            if self.isin(term):
                v = self[term]
            else:
                subterms = term.split('-')
                subterm_sum = self.zeros()
                subterm_cnt = 0
                for subterm in subterms:
                    if self.isin(subterm):
                        subv = self[subterm]
                        subterm_sum += subv
                        subterm_cnt += 1
                if subterm_cnt > 0:
                    v = subterm_sum / subterm_cnt
            if v is not None:
                ret += v
                cnt += 1
        return ret / cnt if cnt > 0 else None
    
def ilsvrc_sem_df(synswords, glove):
    """
        FOR ILSVRC training set
    """
    df_sem = get_glove(synswords, glove)
    df_tr  = base_glove(get_split("train"), glove)
    df_sem = pd.concat([df_sem, df_tr])
    df_sem = df_sem[~df_sem.index.duplicated()]
    return df_sem

def ilsvrc_tree_df(synswords, imcounts, min_photo, max_photo):
    """
        FOR ILSVRC training set
    """
    df = pd.DataFrame({"photo": imcounts, "lemmas":synswords})#.dropna()
    df["sem"]   = df["lemmas"].apply(lambda x:len(x)>0)
    df["train"] = False
    df["train"][get_split("train")]=True
    df["test"]  = (df["photo"]>min_photo) & \
                  (df["photo"]<max_photo) & \
                   df["sem"]
    df["image"]=True
    return df
