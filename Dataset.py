import os
import sys
import numpy as np
import tensorflow as tf
import json
import pickle
import collections
import Vocabulary
import Config
import utils


class BatchInput(collections.namedtuple("BatchInput",
                                        ("key_input", "val_input", "input_lens",
                                         "target_input", "target_output", "output_lens",
                                         "group", "group_lens", "group_cnt",
                                         "target_type", "target_type_lens",
                                         "text", "slens",
                                         "category"))):
    pass

class EPWDataset:
    def __init__(self):
        self.config = Config.config
        if not os.path.exists(self.config.vocab_file):
            pickle.dump(Vocabulary.Vocabulary(), open(self.config.vocab_file, "wb"))
        self.vocab = pickle.load(open(self.config.vocab_file, "rb"))
        utils.print_out("finish reading vocab : {}".format(len(self.vocab.id2word)))
        self.cate2FK = {
            "裙": ["类型", "版型", "材质", "颜色", "风格", "图案", "裙型", "裙下摆", "裙腰型", "裙长", "裙衣长", "裙袖长", "裙领型", "裙袖型", "裙衣门襟",
                  "裙款式"],
            "裤": ["类型", "版型", "材质", "颜色", "风格", "图案", "裤长", "裤型", "裤款式", "裤腰型", "裤口"],
            "上衣": ["类型", "版型", "材质", "颜色", "风格", "图案", "衣样式", "衣领型", "衣长", "衣袖长", "衣袖型", "衣门襟", "衣款式"]}
        for key, val in self.cate2FK.items():
            self.cate2FK[key] = dict(zip(val, range(len(val))))

        self.input_graph = tf.Graph()
        with self.input_graph.as_default():
            proto = tf.ConfigProto()
            proto.gpu_options.allow_growth = True
            self.input_sess = tf.Session(config=proto)
            self.prepare_dataset()

    def get_batch(self, data):
        with self.input_graph.as_default():
            input_initializer, batch = self.make_iterator(data)
            self.input_sess.run(input_initializer)
        return batch

    def next_batch(self, batch):
        with self.input_graph.as_default():
            res = self.input_sess.run(batch)
        return res

    def prepare_dataset(self):
        with self.input_graph.as_default():
            self.dev = self.get_dataset(self.config.dev_file, False)
            self.test = self.get_dataset(self.config.test_file, False)
            self.train = self.get_dataset(self.config.train_file, True)

    def make_iterator(self, data):
        iterator = data.make_initializable_iterator()
        (key_input, val_input, input_lens,
         target_input, target_output, output_lens,
         group, group_lens, group_cnt,
         target_type, target_type_lens,
         text, slens,
         category) = iterator.get_next()
        return iterator.initializer, \
               BatchInput(
                   key_input=key_input,
                   val_input=val_input,
                   input_lens=input_lens,

                   target_input=target_input,
                   target_output=target_output,
                   output_lens=output_lens,

                   group=group,
                   group_lens=group_lens,
                   group_cnt=group_cnt,

                   target_type=target_type,
                   target_type_lens=target_type_lens,

                   text=text,
                   slens=slens,

                   category=category
               )

    def sort(self, cate, lst):
        assert cate in self.cate2FK
        tgt = self.cate2FK[cate]
        return sorted(lst, key=lambda x: tgt.get(x[0], len(tgt) + 1))

    def process_inst(self, line):
        res = {"feats" + suffix: [] for suffix in ['_key', '_val']}
        cate = dict(line['feature'])['类型']
        val_tpe = 1
        feats = self.sort(cate, line['feature'])
        for item in feats:
            res["feats_key"].append(self.vocab.lookup(item[0], 0))
            res["feats_val"].append(self.vocab.lookup(item[1], val_tpe))

        text = [self.vocab.lookup(word, 2) for word in line['desc'].split(" ")]
        slens = len(text)
        res["feats_key_len"] = len(res["feats_key"])

        category = self.vocab.category2id[cate]

        key_input = [self.vocab.lookup("<SENT>", 0)] + res['feats_key']
        val_input = [self.vocab.lookup("<ADJ>", val_tpe)] + res['feats_val']
        input_lens = len(key_input)

        target_input = []
        target_output = []
        output_lens = []

        group = []
        group_lens = []

        target_type = []
        target_type_lens = []

        key_val = list(zip(key_input, val_input))
        for _, segment in line['segment'].items():
            sent = [self.vocab.lookup(w, 2) for w in segment['seg'].split(" ")]
            target_output.append(sent + [self.vocab.end_token])
            target_input.append([self.vocab.start_token] + sent)
            output_lens.append(len(target_output[-1]))

            order = [item[:2] for item in segment['order']]
            if len(order) == 0:
                order = [['<SENT>', '<ADJ>']]
            gid = [key_val.index((self.vocab.lookup(k, 0), self.vocab.lookup(v, val_tpe))) for k, v in order]
            group.append(sorted(gid))
            group_lens.append(len(group[-1]))

            target_type.append([self.vocab.type2id[t] for t in segment['key_type']])
            target_type_lens.append(len(target_type[-1]))

        group_cnt = len(group)

        for item in [target_input, target_output, group, target_type]:
            max_len = -1
            for lst in item:
                max_len = max(max_len, len(lst))
            for idx, lst in enumerate(item):
                if len(lst) < max_len:
                    item[idx] = lst + [0] * (max_len - len(lst))

        return (
            np.array(key_input, dtype=np.int32), np.array(val_input, dtype=np.int32),
            np.array(input_lens, dtype=np.int32),
            np.array(target_input, dtype=np.int32), np.array(target_output, dtype=np.int32),
            np.array(output_lens, dtype=np.int32),
            np.array(group, dtype=np.int32), np.array(group_lens, dtype=np.int32),
            np.array(group_cnt, dtype=np.int32),
            np.array(target_type, dtype=np.int32), np.array(target_type_lens, dtype=np.int32),
            np.array(text, dtype=np.int32), np.array(slens, dtype=np.int32),
            np.array(category, dtype=np.int32),
        )

    def get_dataset(self, filename, train=True):
        def process(line):
            line = json.loads(line.decode())
            return self.process_inst(line)

        dataset = tf.data.TextLineDataset(filename)

        dataset = dataset.map(map_func=lambda x: tf.py_func(lambda y: process(y), [x], Tout=[tf.int32] * 14))

        if train:
            dataset = dataset.shuffle(self.config.shuffle_buffer_size, reshuffle_each_iteration=True)

        def batching_func(x):
            return x.padded_batch(
                self.config.train_batch_size if (train) else self.config.test_batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),
                    tf.TensorShape([None]),
                    tf.TensorShape([]),

                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None]),

                    tf.TensorShape([None, None]),
                    tf.TensorShape([None]),
                    tf.TensorShape([]),

                    tf.TensorShape([None, None]),
                    tf.TensorShape([None]),

                    tf.TensorShape([None]),
                    tf.TensorShape([]),

                    tf.TensorShape([])
                )
            )

        def key_func(p_1, p_2, input_len,
                     p_3, p_4, p_5,
                     p_6, p_7, gcnt,
                     p_8, p_9,
                     p_10, slen,
                     p_11
                     ):
            bucket_id = gcnt # slen // self.config.bucket_width
            return tf.to_int64(tf.minimum(self.config.num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        if train:
            dataset = dataset.apply(
                tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func,
                                                window_size=self.config.train_batch_size))
        else:
            dataset = batching_func(dataset)

        return dataset
