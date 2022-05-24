def TransitionMatrixFh(n):
    M = []
    PartitionIndex = dict()
    par_list = []
    for par in Partitions(n):
        par_list.append(list(par))
    par_list.sort(key=lambda x : len(x))
    for i, la in enumerate(par_list):
        PartitionIndex[str(list(la))] = i
    for la in par_list:
        cnt = [0 for i in range(len(par_list))]
        arr = []
        for i in range(len(la)):
            arr += [i for j in range(la[i])]
        for perm in Permutations(arr):
            comp = set_to_comp(n, Des(perm))
            if str(comp) in PartitionIndex.keys():
                cnt[PartitionIndex[str(comp)]] += 1
        M.append(cnt)
    return Matrix(M).inverse()

def Des(word):
    result = []
    for i in range(1, len(word)):
        if word[i-1] > word[i]:
            result.append(i)
    return result

def set_to_comp(n, S):
    if len(S) == 0:
        return [n]
    comp = [S[0]]
    for i in range(1, len(S)):
        comp.append(S[i]-S[i-1])
    comp.append(n-S[-1])
    return comp

def get_Partitions(N):
    PartitionDict = dict()
    for i in range(1,N+1):
        par_list = []
        for par in Partitions(i):
            par_list.append(list(par))
        par_list.sort(key=lambda x : len(x))
        PartitionDict[i] = par_list
    return PartitionDict

def get_PartitionIndex(N):
    PartitionIndex = dict()
    for i in range(1,N+1):
        par_list = []
        for par in Partitions(i):
            par_list.append(list(par))
        par_list.sort(key=lambda x : len(x))
        PartitionDict = dict()
        for j, par in enumerate(par_list):
            PartitionDict[str(par)] = j
        PartitionIndex[i] = PartitionDict
    return PartitionIndex

def get_PartitionMultiplicity(N):
    PartitionMultiplicity = dict()
    for i in range(1,N+1):
        par_list = []
        for par in Partitions(i):
            par_list.append(list(par))
        par_list.sort(key=lambda x : len(x))
        for par in par_list:
            PartitionMultiplicity[str(i)].append([par.count(k) for k in range(1,i+1)])

