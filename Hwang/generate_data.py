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

def comb_to_shuffle(comb, A, B):
    iterA = iter(A)
    iterB = iter(B)
    return [next(iterA) if i in comb else next(iterB) for i in range(len(A) + len(B))]

def iter_shuffles(lists):
    if len(lists) == 1:
        yield lists[0]
    elif len(lists) == 2:
        for comb in itertools.combinations(range(len(lists[0]) + len(lists[1])), len(lists[0])):
            yield comb_to_shuffle(comb, lists[0], lists[1])
    else:
        length_sum = sum(len(word) for word in lists)
        for comb in itertools.combinations(range(length_sum), len(lists[0])):
            for shuffled in iter_shuffles(lists[1:]):
                yield comb_to_shuffle(comb, lists[0], shuffled)

def cluster_vertices(P):
    n = len(P)
    arr = [0 for i in range(n)]
    k = 0
    for i in range(1,len(P)):
        if P[i-1] != P[i]:
            for j in range(P[i-1], P[i]):
                arr[j] += i
            k += 1
        arr[i] += k
    vertices = [[1]]
    for i in range(1, len(P)):
        if arr[i-1] == arr[i]:
            vertices[-1].append(i+1)
        else:
            vertices.append([i+1])
    return vertices

def get_equiv_classes(P, primitive=True):
    result = []
    perm_list = []
    N = len(P)
    perm_list = dict()

    if primitive:
        iter_words = iter_shuffles(cluster_vertices(P))
    else:
        iter_words = itertools.permutations(range(1,N+1))

    for perm in iter_words:
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

def generate_data(DIR_PATH, N=7, primitive = True, connected=False, extended=True):
    n = 0
    Ms = []
    XPs = []
    for P in iter_UIO(N, connected):
        equiv_list = get_equiv_classes(P, primitive)
        for equiv_class in equiv_list:
            noDes = get_noDesWords(P, equiv_class)
            pars = K_H(P, equiv_class)
            noDes_words_along_sinks = cluster_words_along_sink(P, noDes)
            pars_along_length = cluster_partitions_along_length(P, pars)
            for word_list in noDes_words_along_sinks:
                Ms.append(make_block_diagonal_sparse_matrix(P, word_list))
            for XP in pars_along_length:
                temp = [int(val) for val in list(XP)]
                XPs.append(temp)
            if extended and len(noDes) > 1:
                Ms.append(make_block_diagonal_sparse_matrix(P, noDes))
                temp = [int(val) for val in list(pars)]
                XPs.append(temp)
    
    # with open(DIR_PATH+f"XP_{N}_onehot.json", 'w') as f:
    #     json.dump(XPs, f)
    
    XP_mult = [[0 for i in range(N)] for j in range(len(XPs))]
    for n in range(1, N+1):
        XP_n = []
        for i in range(len(XPs)):
            XP_n.append(0)
            for k in range(len(XPs[i])):
                XP_n[-1] += XPs[i][k] * PartitionMultiplicity[str(N)][k][n-1]
            XP_mult[i][n-1] = XP_n[-1]
    if extended:
        cnt = 0
        m = len(XPs)
        while cnt < m:
            k = np.random.randint(m)
            if not 0 in XP_mult[k]:
                continue
            mats = [Ms[k]]
            arr = np.array(XP_mult[k])
            pos = list(np.where(arr == 0)[0])
            for p in pos:
                while True:
                    k = np.random.randint(m)
                    if XP_mult[k][p] > 0:
                        mats.append(Ms[k])
                        arr += np.array(XP_mult[k])
                        break
            k = np.random.randint(m)
            mats.append(Ms[k])
            arr += np.array(XP_mult[k])
            Ms.append(sp.block_diag(mats))
            temp = [int(val) for val in list(arr)]
            XP_mult.append(temp)
            cnt += 3
    
    for n, mat in enumerate(Ms):
        sp.save_npz(DIR_PATH+f"graph_{n:05d}.npz", mat)
    with open(DIR_PATH+f"XP_{N}_multiplicity.json", 'w') as f:
        json.dump(XP_mult, f)

with open("PartitionIndex.json", "r") as f:
    PartitionIndex = json.load(f)
with open("TransitionMatrix.json", "r") as f:
    TMs = json.load(f)
with open("Partitions.json", "r") as f:
    Partitions = json.load(f)
with open("PartitionMultiplicity.json", "r") as f:
    PartitionMultiplicity = json.load(f)

