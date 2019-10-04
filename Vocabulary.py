import os
import sys
import json
import numpy as np
import pickle
import Config

class Vocabulary:
	def __init__(self):
		self.config = Config.config

		category = list(pickle.load(open(self.config.category_file, "rb")))
		featCate = list(pickle.load(open(self.config.feat_key_file, "rb")))
		featVal = list(pickle.load(open(self.config.feat_val_file, "rb")))
		cateFK2val = pickle.load(open(self.config.cateFK2val_file, "rb"))
		self.cateFK2val = cateFK2val

		self.id2category = category
		self.category2id = dict(zip(self.id2category, range(len(self.id2category))))

		self.id2featCate = ["<MARKER>", "<SENT>"] + featCate
		self.featCate2id = dict(zip(self.id2featCate, range(len(self.id2featCate))))

		self.id2type = ["<GENERAL>"] + featCate
		self.type2id =dict(zip(self.id2type, range(len(self.id2type))))

		self.id2featVal = ["<S>", "<ADJ>"] + featVal
		self.featVal2id = dict(zip(self.id2featVal, range(len(self.id2featVal))))

		self.id2word = ["<S>", "</S>", 0] + [0] * len(featVal)
		self.id2vec = [0] * (3 + len(featVal))
		nxt = 3
		with open(self.config.wordvec_file) as file:
			for _ in range(self.config.skip_cnt):
				file.readline()
			for line in file:
				line = line.split(" ")
				word = line[0]
				vec = [eval(i) for i in line[1:]]
				if word in featVal:
					self.id2word[nxt] = word
					self.id2vec[nxt] = vec
					nxt += 1
				elif word == "<UNK>":
					self.id2word[2] = "<UNK>"
					self.id2vec[2] = vec
				else:
					self.id2word.append(word)
					self.id2vec.append(vec)
		for val in featVal:
			if val not in self.id2word:
				self.id2word.append(val)
				self.id2vec[nxt] = list(np.random.uniform(low=-0.1, high=0.1, size=(self.config.word_dim, )))
				nxt += 1
		assert nxt == len(featVal) + 3
		self.keywords_cnt = nxt
		fcnt = 2
		if "<UNK>" not in self.id2word:
			self.id2word[2] = "<UNK>"
			fcnt += 1
		for i in range(fcnt):
			self.id2vec[i] = list(np.random.uniform(low=-0.1, high=0.1, size=(self.config.word_dim, )))
		self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

		self.table = [self.featCate2id, self.featVal2id, self.word2id, self.type2id]

		self.start_token = 0
		self.end_token = 1

	def lookup(self, word, tpe):
		"""
		:param word:
		:param tpe: 0 for featCate
					1 for featVal
					2 for word
					3 for type
		:return:
		"""
		if tpe == 2:
			return self.table[tpe].get(word, self.table[tpe]["<UNK>"])
		else:
			return self.table[tpe][word]
