
def max_permute(s):
    words = s.split()
    for i, word in enumerate(words):
        chars = list(word)
        for j, c in enumerate(chars):
            new_place = j
            # iterate over word, find place for this char
            for k, cnew in zip(range(len(chars)), chars):
                # if new char is not old char, and new char is not in old place
                # and cur char also not in new place, swap
                if cnew != c and cnew != word[j] and c != word[k]:
                    new_place = k
                    break
            chars[new_place], chars[j] = chars[j], chars[new_place]
        words[i] = ''.join(chars)
    return ' '.join(words)


def cnt_unchanged(orig, perm):
    return sum([1 if c_orig == c_perm else 0
                for word_orig, word_perm in zip(orig.split(), perm.split())
                for c_orig, c_perm in zip(word_orig, word_perm)])


if __name__ == '__main__':
    s = input("Enter string to permute:")
    s_perm = max_permute(s)
    cnt = cnt_unchanged(s, s_perm)
    print(s_perm, cnt, sep='\n')
