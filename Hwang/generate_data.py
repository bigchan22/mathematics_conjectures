import json
import itertools
import numpy as np
import networkx as nx
import scipy.sparse as sp

def generate_UIO(n, connected=False):
    if connected == False:
        k = 1
    else:
        k = 2
    seq = [i+k for i in range(n)]          
    seq[n-1] = n
    list_UIO = [copy(seq)]
    while seq[0] < n:
        for i in range(n-1):
            if seq[i] < seq[i+1]:
                seq[i] += 1
                for j in range(i):
                    seq[j] = j+k
                break
        list_UIO.append(copy(seq))
    return list_UIO

def iter_UIO(n, connected=False):
    if connected == False:
        k = 1
    else:
        k = 2
    seq = [i+k for i in range(n)]          
    seq[n-1] = n
    seq[0] -= 1
    while seq[0] < n:
        for i in range(n-1):
            if seq[i] < seq[i+1]:
                seq[i] += 1
                for j in range(i):
                    seq[j] = j+k
                break
        yield seq

def get_equiv_classes(P):
    result = []
    perm_list = []
    N = len(P)
    N_list = [i for i in range(1, N+1)]
    perm_list = dict()

    for perm in itertools.permutations(N_list):
        perm = list(perm)
        if str(perm) in perm_list.keys():
            continue
        perm_list[str(perm)] = True
        queue = [perm]
        for perm in queue:
            word_list = get_equiv_words(P, perm)
            for word in word_list:
                if str(word) in perm_list.keys():
                    continue
                perm_list[str(word)] = True
                queue.append(word)
        result.append(queue)
    return result

def get_equiv_words(P, word):
    word_list = []
    for i in range(len(word)-1):
        if is_compatible(P, word[i], word[i+1]):
            word_list.append(word[:i]+[word[i+1],word[i]]+word[i+2:])
    for i in range(1, len(word)-1):
        if is_P_less(P, word[i-1], word[i]) and not is_compatible(P, word[i-1], word[i+1]) and not is_compatible(P, word[i], word[i+1]):
            word_list.append(word[:i-1]+[word[i+1],word[i-1],word[i]]+word[i+2:])
        elif is_P_less(P, word[i], word[i+1]) and not is_compatible(P, word[i-1], word[i]) and not is_compatible(P, word[i-1], word[i+1]):
            word_list.append(word[:i-1]+[word[i],word[i+1],word[i-1]]+word[i+2:])
    return word_list

def is_compatible(P, a, b):
    if P[a-1] < b or P[b-1] < a:
        return True
    return False

def is_P_less(P, a, b):
    if P[a-1] < b:
        return True
    return False

def P_Des(P, word):
    Des = []
    for i in range(1, len(word)):
        if is_P_less(P, word[i], word[i-1]):
            Des.append(i)
    return Des

def set_to_comp(n, S):
    if len(S) == 0:
        return [n]
    comp = [S[0]]
    for i in range(1, len(S)):
        comp.append(S[i]-S[i-1])
    comp.append(n-S[-1])
    return comp

def K_H(P, word_list):
    n = len(P)
    n_str = str(n)
    K = np.zeros(len(PartitionIndex[n_str]))
    for word in word_list:
        comp = set_to_comp(len(word), P_Des(P, word))
        if str(comp) in PartitionIndex[n_str].keys():
            K[PartitionIndex[n_str][str(comp)]] += 1
    return np.matmul(K, np.array(TMs[n_str]))

def get_noDesWords(P, word_list):
    result = []
    for word in word_list:
        if P_Des(P, word) == []:
            result.append(word)
    return result

def get_sink_number(P, word):
    n = 1
    for i in range(1, len(word)):
        chk = 0
        for j in range(i):
            if is_compatible(P, word[i], word[j]) == 0:
                chk = 1
                break
        if chk == 0:
            n += 1
    return n

def cluster_words_along_sink(P, word_list):
    words_sink = [[] for i in range(len(P))]
    for word in word_list:
        words_sink[get_sink_number(P, word)-1].append(word)
    result = []
    for temp in words_sink:
        if temp != []:
            result.append(temp)
    return result

def cluster_partitions_along_length(P, K):
    n_str = str(len(P))
    partitions_length = [np.zeros(len(Partitions[n_str]), dtype=int) for i in range(len(P))]
    length_flag = [False for i in range(len(P))]
    for par in Partitions[n_str]:
        ind = PartitionIndex[n_str][str(par)]
        if K[ind] > 0:
            partitions_length[len(par)-1][ind] = K[ind]
            length_flag[len(par)-1] = True
    result = []
    for i, TF in enumerate(length_flag):
        if TF:
            result.append(partitions_length[i])
    return result

def make_sparse_matrix(P, word):
    n = len(P)
    row = []
    col = []
    data = []
    for i in range(1,n):
        for j in range(i):
            if is_compatible(P, word[i], word[j]) == False:
                row.append(word[i]-1)
                col.append(word[j]-1)
                data.append(1)
    return sp.coo_matrix((data, (row,col)), shape=(n,n))

def make_block_diagonal_sparse_matrix(P, word_list):
    mats = []
    for word in word_list:
        mats.append(make_sparse_matrix(P, word))
    return sp.block_diag(mats)

def generate_data(DIR_PATH, N=7, connected=False):
    n = 0
    XPs = []
    for P in iter_UIO(N, connected):
        equiv_list = get_equiv_classes(P)
        for equiv_class in equiv_list:
            noDes_words_along_sinks = cluster_words_along_sink(P, get_noDesWords(P, equiv_class))
            pars_along_length = cluster_partitions_along_length(P, K_H(P, equiv_class))
            for word_list in noDes_words_along_sinks:
                M = make_block_diagonal_sparse_matrix(P, word_list)
                sp.save_npz(DIR_PATH+f"graph_{n:04d}.npz", M)
                n += 1
            for XP in pars_along_length:
                temp = [int(val) for val in list(XP)]
                XPs.append(temp)
    with open(DIR_PATH+f"XP_{N}_onehot.json", 'w') as f:
        json.dump(XPs, f)
    for n in range(1, N+1):
        with open(DIR_PATH+f"XP_{N}_{n}.json", 'w') as f:
            XP_n = []
            for i in range(len(XPs)):
                XP_n.append(0)
                for k in range(len(XPs[i])):
                    XP_n[-1] += XPs[i][k] * PartitionMultiplicity[str(N)][k][n-1]
            json.dump(XP_n, f)
                

with open("PartitionIndex.json", "r") as f:
    PartitionIndex = json.load(f)
with open("TransitionMatrix.json", "r") as f:
    TMs = json.load(f)
with open("Partitions.json", "r") as f:
    Partitions = json.load(f)
with open("PartitionMultiplicity.json", "r") as f:
    PartitionMultiplicity = json.load(f)

