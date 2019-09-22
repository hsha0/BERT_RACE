from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import glob
import json

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

class RaceExample(object):

    def __init__(self, id, article, question, answer, options):
        self.id = id
        self.article = article
        self.question = question
        self.answer = answer
        self.options = options

def print_example(example):
    print('id:', example.id)
    print('article:', example.article)
    print()
    print('question:', example.question)
    print('answer:', example.answer)
    print('options:', example.options)
    print()


class InputFeature(object):

    def __init__(self,
                 unique_id,
                 example_index,
                 choices_features,
                 answer_id
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.choices_features = list(choices_features)
        self.answer = answer_id



def read_race_examples(data_dir):

    def _read_race_examples(filename):
        examples = []
        with open(filename) as json_file:
            instance = json.load(json_file)
            for i in range (len(instance['answers'])):
                example = RaceExample(instance['id']+'_'+str(i), instance['article'], instance['questions'][i],
                                      instance['answers'][i], instance['options'][i])
                #print_example(example)
                examples.append(example)
        return examples

    if not os.path.exists(data_dir):
        sys.exit(data_dir, "doesn't exist.")
    pre_dir = os.getcwd()
    os.chdir(data_dir)
    file_list = sorted(glob.glob('*.txt'), key=lambda x:int(x[:-4]))
    examples = []
    for file in file_list:
        examples.extend(_read_race_examples(file))
    os.chdir(pre_dir)

    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_article = tokenizer.tokenize(example.article)
    tokens_question = tokenizer.tokenize(example.question)
    choices_features = []

    for i in range(len(example.options)):
        tokens_option = tokenizer.tokenize(example.options[i])
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_option:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_question:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)


        max_length = max_seq_length - 1 - len(tokens)
        tokens_article = tokens_article[:max_length]

        for token in tokens_article:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length



        feature = {}
        feature['input_ids'] = input_ids
        feature['input_mask'] = input_mask
        feature['segment_ids'] = segment_ids

        choices_features.append(feature)

    answer_id = label_map[example.answer]
    feature = InputFeature(
        unique_id=example.id,
        example_index=ex_index,
        choices_features=choices_features,
        answer_id=answer_id
    )

    return feature


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    features = []
    for i in range(len(examples)):
        feature = convert_single_example(i, examples[i], label_list, max_seq_length, tokenizer)
        features.append(feature)
    return features

def create_model (bert_config, is_training, feature, labels, num_labels, use_one_hot_embeddings):

    for i in range(len(feature.choices_features)):
        choices_feature = feature.choices_features[i]
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=choices_feature['input_ids'],
            input_mask=choices_feature['input_mask'],
            token_type_ids=choices_feature['segment_ids'],
            use_one_hot_embeddings=use_one_hot_embeddings)

        output_layer = model.get_pooled_output()
        print(output_layer)

def main():
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    examples = read_race_examples(FLAGS.data_dir)
    features = convert_examples_to_features(examples, ['A','B','C','D'], 512, tokenizer, FLAGS.output_dir)


if __name__ == '__main__':
    main()


