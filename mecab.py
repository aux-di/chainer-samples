# -*- coding: utf-8 -*-

import MeCab

text = "pythonでmecabを使ってみる"

m = MeCab.Tagger("-Ochasen")
print m.parse(text)

m = MeCab.Tagger("-Owakati")
print m.parse(text)
