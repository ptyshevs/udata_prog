import random as rd


def permute_chars(s):
    words = s.split()
    for i, word in enumerate(words):
        word_chars = list(word)
        if len(word) < 3:
            continue
        n_perm = len(word) // 2
        if n_perm > 1:
            n_perm = 1
        for j in range(n_perm):
            c_from = rd.choice(word[1:-1])
            from_i = word.index(c_from)
            word_chars[from_i], word_chars[from_i - 1] = word_chars[from_i - 1], word_chars[from_i]
        words[i] = ''.join(word_chars)
    return ' '.join(words)


if __name__ == '__main__':
    s = input("Enter string to permute:")
    # s = "Написать программу, которая перемешивает символы в словах заданного текста. При этом текст после перестановок должен оставаться читабельным."
    print(permute_chars(s))
