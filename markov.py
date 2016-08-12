# -*- coding: utf-8 -*-
import random
import MeCab

def wakati(text):
    t = MeCab.Tagger('-Owakati')
    m = t.parse(text)
    result = m.rstrip(' \n').split(' ')
    return result

if __name__ == '__main__':
    filename = 'txt/raw-text.txt'
    src = open(filename, 'r').read()
    wordlist = wakati(src)

    # Create table of Markov Chain
    markov = {}
    w1 = ''
    w2 = ''
    w3 = ''
    for word in wordlist:
        if w1 and w2 and w3:
            if (w1, w2, w3) not in markov:
                markov[(w1, w2, w3)] = []
            markov[(w1, w2, w3)].append(word)
        w1, w2, w3 = w2, w3, word

    # Generate Sentence
    count = 0
    sentence = ''
    w1, w2, w3 = random.choice(markov.keys())
    while count < len(wordlist):
        tmp = random.choice(markov[(w1, w2, w3)])
        sentence += tmp
        w1, w2, w3 = w2, w3, tmp
        count += 1

    print sentence
