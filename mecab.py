# -*- coding: utf-8 -*-

import MeCab

# text = "pythonでmecabを使ってみる"
# m = MeCab.Tagger("-Ochasen")
# print m.parse(text)
# m = MeCab.Tagger("-Owakati")
# print m.parse(text)

m = MeCab.Tagger("-Owakati")

fin = open('txt/raw-text.txt', 'r')
fout = open('txt/words.txt', 'w')

for line in fin:
    fout.write(m.parse(line))

fin.close()
fout.close()
