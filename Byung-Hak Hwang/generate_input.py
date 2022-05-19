import json
import numpy as np
import itertools

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


with open("PartitionIndex.json", "r") as f:
    PartitionIndex = json.load(f)
with open("TransitionMatrix.json", "r") as f:
    TMs = json.load(f)