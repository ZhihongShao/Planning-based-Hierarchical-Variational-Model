import os
import sys
import numpy as np
import json
import argparse
import Config
import pickle

def import_lib():
	global Dataset, utils, tf, device_lib, PHVM, Dataset, model_utils
	import tensorflow as tf
	from tensorflow.python.client import device_lib
	import utils

	import Dataset

	from Models import PHVM
	from Models import model_utils

def dump(texts, filename):
	file = open(filename, "w")
	for inst in texts:
		lst = []
		for sent in inst:
			sent = " ".join(sent)
			lst.append({'desc': sent})
		file.write(json.dumps(lst, ensure_ascii=False) + "\n")
	file.close()

def infer(model, dataset, data):
	config = Config.config
	brand_set = pickle.load(open(config.brand_set_file, "rb"))
	vocab = dataset.vocab
	batch = dataset.get_batch(data)
	res = []
	while True:
		try:
			batchInput = dataset.next_batch(batch)

			output = model.infer(batchInput)
			_output = []
			for inst_id, inst in enumerate(output):
				sents = []
				dup = set()
				for beam in inst:
					sent = []
					for wid in beam:
						if wid == dataset.vocab.end_token:
							break
						elif wid == dataset.vocab.start_token:
							continue
						sent.append(vocab.id2word[wid] if vocab.id2word[wid] not in brand_set else "BRAND")
					if str(sent) not in dup:
						dup.add(str(sent))
						sents.append(sent)
				_output.append(sents)
			res.extend(_output)
		except tf.errors.OutOfRangeError:
			break
	return res

def evaluate(model, dataset, data):
	batch = dataset.get_batch(data)
	tot_loss = 0
	tot_cnt = 0
	while True:
		try:
			batchInput = dataset.next_batch(batch)
			global_step, loss = model.eval(batchInput)
			slens = batchInput.slens
			tot_cnt += len(slens)
			tot_loss += loss * len(slens)
		except tf.errors.OutOfRangeError:
			break
	return tot_loss / tot_cnt

def _train(model_name, model, dataset, summary_writer, init):
	best_loss = 1e20
	batch = dataset.get_batch(dataset.train)
	epoch = init['epoch']
	worse_step = init['worse_step']
	logger.info("epoch {}".format(epoch))
	if model.get_global_step() > config.num_training_step or worse_step > model.early_stopping:
		return
	while True:
		try:
			batchInput = dataset.next_batch(batch)
			global_step, loss, train_summary = model.train(batchInput)

			if global_step % config.steps_per_stat == 0:
				summary_writer.add_summary(train_summary, global_step)
				summary_writer.flush()
				logger.info("{} step : {:.5f}".format(global_step, loss))
		except tf.errors.OutOfRangeError:
			eval_loss = evaluate(model, dataset, dataset.dev)
			utils.add_summary(summary_writer, global_step, "dev_loss", eval_loss)
			logger.info("dev loss : {:.5f}".format(eval_loss))

			if eval_loss < best_loss:
				worse_step = 0
				best_loss = eval_loss
				prefix = config.checkpoint_dir + "/" + model_name + config.best_model_dir
				model.best_saver.save(model.sess, prefix + "/best_{}".format(epoch), global_step=global_step)
			else:
				worse_step += 1
				prefix = config.checkpoint_dir + "/" + model_name + config.tmp_model_dir
				model.tmp_saver.save(model.sess, prefix + "/tmp_{}".format(epoch), global_step=global_step)
			if global_step > config.num_training_step or worse_step > model.early_stopping:
				break
			else:
				batch = dataset.get_batch(dataset.train)
			epoch += 1
			logger.info("\nepoch {}".format(epoch))

def train(model_name, restore=True):
	import_lib()
	global config, logger
	config = Config.config
	dataset = Dataset.EPWDataset()
	dataset.prepare_dataset()
	logger = utils.get_logger(model_name)

	model = PHVM.PHVM(len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), len(dataset.vocab.id2word),
					  len(dataset.vocab.id2category),
					  key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec,
					  type_vocab_size=len(dataset.vocab.id2type))
	init = {'epoch': 0, 'worse_step': 0}
	if restore:
		init['epoch'], init['worse_step'], model = model_utils.restore_model(model,
											config.checkpoint_dir + "/" + model_name + config.tmp_model_dir,
											config.checkpoint_dir + "/" + model_name + config.best_model_dir)
	config.check_ckpt(model_name)
	summary = tf.summary.FileWriter(config.summary_dir, model.graph)
	_train(model_name, model, dataset, summary, init)
	logger.info("finish training {}".format(model_name))

def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == 'true')
	parser.add_argument("--cuda_visible_devices", type=str, default='0,1,2,3')
	parser.add_argument("--train", type="bool", default=True)
	parser.add_argument("--restore", type="bool", default=False)
	parser.add_argument("--model_name", type=str, default="PHVM")
	args = parser.parse_args(sys.argv[1:])
	return args

def main():
	args = get_args()

	if args.train:
		train(args.model_name, args.restore)
	else:
		import_lib()
		dataset = Dataset.Dataset()
		model = PHVM.PHVM(len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), len(dataset.vocab.id2word),
						  len(dataset.vocab.id2category),
						  key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec,
						  type_vocab_size=len(dataset.vocab.id2type))

		best_checkpoint_dir = config.checkpoint_dir + "/" + args.model_name + config.best_model_dir
		tmp_checkpoint_dir = config.checkpoint_dir + "/" + args.model_name + config.tmp_model_dir
		model_utils.restore_model(model, best_checkpoint_dir, tmp_checkpoint_dir)

		dataset.prepare_dataset()
		texts = infer(model, dataset, dataset.test)
		dump(texts, config.result_dir + "/{}.json".format(args.model_name))
		utils.print_out("finish file test")

if __name__ == "__main__":
	main()
