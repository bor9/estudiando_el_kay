import string

letters = string.ascii_lowercase
N_max = len(letters)

N = 10

N = min(N, N_max)

# generate list of all possible words with the N first letters of the alphabet
words = []
for i in range(N):
    a = letters[i]
    for j in range(N):
        b = letters[j]
        for k in range(N):
            c = letters[k]
            for l in range(N):
                words.append(a + b + c + letters[l])


l_all_equal = []
l_three_equal = []
l_two_equal_two_equal = []
l_two_equal_two_distinct = []
l_all_different = []
for i in range(N ** 4):
    set_words_i = set(words[i])
    if len(set_words_i) == 1:
        # all distinct
        l_all_equal.append(words[i])
    elif len(set_words_i) == 4:
        # all equal
        l_all_different.append(words[i])
    elif len(set_words_i) == 3:
        # two equal and two distinct
        l_two_equal_two_distinct.append(words[i])
    else:
        # two equal and two equal or three equal
        if words[i].count(words[i][0]) == 2:
            l_two_equal_two_equal.append(words[i])
        else:
            l_three_equal.append(words[i])


n_all_equal = len(l_all_equal)
n_three_equal = len(l_three_equal)
n_two_equal_two_equal = len(l_two_equal_two_equal)
n_two_equal_two_distinct = len(l_two_equal_two_distinct)
n_all_different = len(l_all_different)


print("N = {}, n_all_equal = {}".format(N, n_all_equal))
print("N(N-1)(N-2)(N-3) = {}, n_all_different = {}".format(N*(N-1)*(N-2)*(N-3), n_all_different))
print("6N(N-1)(N-2) = {}, n_two_equal_two_distinct = {}".format(6*N*(N-1)*(N-2), n_two_equal_two_distinct))
print("3N(N-1) = {}, n_two_equal_two_equal = {}".format(3*N*(N-1), n_two_equal_two_equal))
print("4N(N-1) = {}, n_three_equal = {}".format(4*N*(N-1), n_three_equal))


print(n_all_equal+n_three_equal+n_two_equal_two_equal+n_two_equal_two_distinct+n_all_different)
