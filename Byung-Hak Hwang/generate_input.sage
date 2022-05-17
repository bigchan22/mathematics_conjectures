R = QQ['t']
t = R.0
Sym = SymmetricFunctions(R)
s = Sym.schur()
e = Sym.elementary()
p = Sym.powersum()
h = Sym.homogeneous()
m = Sym.monomial()
f = e.dual_basis()
q = p.dual_basis()

QSym = QuasiSymmetricFunctions(R)
M = QSym.M()
F = QSym.F()
QS = QSym.QS()

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

def get_equiv_classes(P):
    result = []
    perm_list = []
    N = len(P)
    for perm in Permutations(N):
        perm = list(perm)
        if perm in perm_list:
            continue
        queue = [perm]
        for perm in queue:
            word_list = get_equiv_words(P, perm)
            for word in word_list:
                if word in queue:
                    continue
                queue.append(word)
        result.append(queue)
        perm_list += queue
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
    K = 0
    for word in word_list:
        K += F(set_to_comp(len(word), P_Des(P, word)))
    return h(K.to_symmetric_function())