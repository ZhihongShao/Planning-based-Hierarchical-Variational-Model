import os
import sys
import jieba
import pickle
import argparse
import gensim

class MySentences:
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		with open(self.filename, "r") as file:
			for line in file:
				yield [word for word in line.strip().split(" ") if len(word) > 0]

def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == "true")
	parser.add_argument("--corpus", type=str, help="specify the src file")
	parser.add_argument("--result_dir", type=str, default="../result/", help="specify the storage location") 
	parser.add_argument("--model_name", type=str, default="/model", help="specify the filename of the model")
	parser.add_argument("--wordvec_name", type=str, default="/wordvec.txt", help="specify the filename of the word vector")
	parser.add_argument("--isEnglish", type="bool", default=True, help="whether the corpus is English or not")
	parser.add_argument("--min_count", type=int, default=5)
	parser.add_argument("--iter", type=int, default=5)
	parser.add_argument("--window", type=int, default=5)
	parser.add_argument("--word_dim", type=int, default=300)
	args, _ = parser.parse_known_args()
	if not os.path.exists(args.result_dir):
		os.mkdir(args.result_dir)
	return args

def train(args):
	corpus = MySentences(args.corpus)
	model = gensim.models.Word2Vec(corpus, size=args.word_dim, window=args.window, min_count=args.min_count, iter=args.iter)
	model.wv.save_word2vec_format(args.result_dir + "/" + args.wordvec_name)
	model.save(args.result_dir + "/" + args.model_name)

if __name__ == "__main__":
	args = get_args()
	train(args)
