import numpy as np
import pandas as pd
import json
import os
import re
import random
import tensorflow as tf
import spacy
import bert


class Node(object):
    def __init__(self, data=None, parent=None, children=[], ID=None, data_type=None, is_impossible=None):
        self.id = ID
        self.data = data
        self.type = data_type
        self.parent = parent
        self.children = [child for child in children]
        self.is_impossible = is_impossible

    def __len__(self):
        return len(self.children)

    def __getitem__(self, idx):
        return self.children[idx]


class DataGenerator(object):

    def __init__(self, fname=None, sqaud=False, batch_size=1200, neg_rate=1., pos_aug=0, noise=0., shuffle=True):
        assert int(pos_aug) >= 0
        assert noise >= 0 and noise < 1
        assert neg_rate > 0 and neg_rate <= 1
        assert int(batch_size) > 0
        assert type(shuffle) == bool

        self.children = []
        self.batch_size = int(batch_size)
        self.data_noise = noise
        self.shuffle = shuffle
        self.aug_rate = int(pos_aug)
        self.neg_sample_rate = neg_rate
        self.end_epoch = False
        self.count = 0
        random.seed()

        if fname is not None:
            if sqaud is False:
                self.read_json(fname)
            else:
                self.load_squad(fname)

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, idx):
        return self._get_batch()

    def read_json(self, fname):
        def dict_to_node(child_dict, parent_node=None):
            child_node = Node(ID=child_dict["id"], data=child_dict["data"], data_type=child_dict["type"],
                              parent=parent_node, is_impossible=child_dict["is_impossible"])
            for child in child_dict["children"]:
                child_node.children.append(
                    dict_to_node(child, parent_node=child_node))
            return child_node

        self.children = []

        with open(fname, 'r') as json_file:
            json_dict = json.loads(json_file.read())

        children = json_dict["root"]
        for child in children:
            self.children.append(dict_to_node(child))
        self.data_size = self._get_data_size()
        self._init_iter()

    def write_json(self, fname):
        def node_to_dict(node):
            node_dict = {"id": node.id, "data": node.data, "type": node.type,
                         "is_impossible": node.is_impossible, "children": []}
            for child in node.children:
                node_dict["children"].append(node_to_dict(child))
            return node_dict

        data_tree_dict = {"root": []}
        for child in self.children:
            assert child.type == "article"
            data_tree_dict["root"].append(node_to_dict(child))

        with open(fname, 'w') as json_file:
            json.dump(data_tree_dict, json_file)

    def load_squad(self, fname):
        with open(fname) as data:
            squad = json.load(data)["data"]

        nlp = spacy.load("en_core_web_md")

        for a_id, article in enumerate(squad):
            print(f"A - {a_id}")
            article_node = Node(ID=a_id, data_type="article")
            self.children.append(article_node)

            for p_id, paragraph in enumerate(article['paragraphs']):
                clean_paragraph = bert.albert_tokenization.preprocess_text(
                    paragraph["context"])
                nlp_text = nlp(clean_paragraph)
                paragraph_data = [sent.text for sent in nlp_text.sents]
                paragraph_node = Node(
                    data=paragraph_data, parent=article_node, ID=p_id, data_type="paragraph")
                article_node.children.append(paragraph_node)

                for question in paragraph["qas"]:
                    question_node = Node(data=question["question"], parent=paragraph_node,
                                         data_type="question", is_impossible=False)
                    paragraph_node.children.append(question_node)

        self.data_size = self._get_data_size()
        self._init_iter()

    def train_test_split(self, test_size=0.25):
        assert test_size > 0 and test_size < 1
        random.shuffle(self.children)
        train = DataGenerator(batch_size=self.batch_size, pos_aug=self.aug_rate,
                              noise=self.data_noise, shuffle=self.shuffle, neg_rate=self.neg_sample_rate)
        test = DataGenerator(batch_size=self.batch_size, pos_aug=self.aug_rate,
                             noise=self.data_noise, shuffle=self.shuffle, neg_rate=self.neg_sample_rate)
        test_len = int(test_size * len(self.children))
        train.children = self.children[test_len:]
        test.children = self.children[:test_len]
        train.data_size = train._get_data_size()
        test.data_size = test._get_data_size()
        train._init_iter()
        test._init_iter()
        return train, test

    def transform_data(self, transform_func):
        def transform_child_data(node, trans_func=transform_func):
            if node.data is not None:
                node.data = trans_func(node.data)
                if node.type == "question":
                    node.data = node.data[0]
            for child in node.children:
                transform_child_data(child)

        for i, child in enumerate(self.children):
            print(f'ARTICLE {i}')
            transform_child_data(child)

    def _init_iter(self):
        self.end_epoch = False
        self.active_article_idx = -1
        self.active_paragraph_idx = -1
        self.active_question_idx = -1

        self.active_article = None
        self.active_paragraph = None
        self.active_question = None

        self._set_next_active_article()
        self._set_next_active_paragraph()
        self._set_next_active_question()

        self._set_paragraph_list()

        self.PAR_LIST_IDX = 0

    def _create_augmented_pos_paragraph(self):
        data_copy = self.active_paragraph.data.copy()
        aug_par = Node(data=data_copy, parent=self.active_paragraph.parent, children=self.active_paragraph.children,
                       ID=self.active_paragraph.id, data_type=self.active_paragraph.type, is_impossible=self.active_paragraph.is_impossible)
        if self.shuffle:
            random.shuffle(aug_par.data)
        for i, sent in enumerate(aug_par.data):
            noise = np.random.normal(0, self.data_noise, len(sent))
            sent = sent + noise
            aug_par.data[i] = np.clip(sent, a_min=-1., a_max=1.).tolist()
        return aug_par

    def _create_paragraph_list(self, par_list):
        par_list.remove(self.active_paragraph)
        num_false_par = int(len(par_list) * self.neg_sample_rate)
        random.shuffle(par_list)
        par_list = par_list[:num_false_par]
        par_list.append(self.active_paragraph)
        if self.active_question.is_impossible is False:
            for _ in range(self.aug_rate):
                aug_par = self._create_augmented_pos_paragraph()
                par_list.append(aug_par)
        random.shuffle(par_list)
        return par_list

    def _set_paragraph_list(self):
        if self.end_epoch is True:
            return
        par_list = self.active_article.children.copy()
        self.paragraph_list = self._create_paragraph_list(par_list)

    def _set_next_active_article(self):
        if self.active_article == self.children[-1]:
            self.end_epoch = True
            return
        self.active_article_idx += 1
        self.active_article = self.children[self.active_article_idx]

    def _set_next_active_paragraph(self):
        if self.end_epoch is True:
            return
        elif self.active_paragraph == self.active_article[-1]:
            self._set_next_active_article()
            self.active_paragraph_idx = -1
            self.active_paragraph = None
            self._set_next_active_paragraph()
        else:
            self.active_paragraph_idx += 1
            self.active_paragraph = self.active_article[self.active_paragraph_idx]

    def _set_next_active_question(self):
        if self.end_epoch is True:
            return
        elif self.active_question == self.active_paragraph[-1]:
            self._set_next_active_paragraph()
            self.active_question_idx = -1
            self.active_question = None
            self._set_next_active_question()
        else:
            self.active_question_idx += 1
            self.active_question = self.active_paragraph[self.active_question_idx]
            self._set_paragraph_list()

    def _get_data_size(self):
        size = 0
        for article in self.children:
            for paragraph in article.children:
                num_questions = len(paragraph.children)
                num_possible_questions = self._get_possible_questions(
                    paragraph)
                num_impossible_questions = num_questions - num_possible_questions
                num_paragraphs = len(article.children)
                num_false_paragraphs = int(
                    (num_paragraphs - 1) * self.neg_sample_rate)
                size += ((num_false_paragraphs * num_questions) +
                         (num_possible_questions * (self.aug_rate + 1)) + num_impossible_questions)
        return size

    @staticmethod
    def _get_possible_questions(paragraph_node):
        count = 0
        for question in paragraph_node.children:
            if question.is_impossible is False:
                count += 1
        return count

    def _get_batch(self):
        X_question_batch = []
        X_paragraph_batch = []
        y_batch = []
        self.count += 1
        for _ in range(self.batch_size):
            X_question, X_paragraph, y = self._get_next_datapoint()
            X_question_batch.append(X_question)
            X_paragraph_batch.append(X_paragraph)
            y_batch.append(y)
            self._update_iter()
            if self.end_epoch is True:
                self._on_epoch_end()
                break

        X_question_batch = np.array(X_question_batch)
        X_paragraph_batch = tf.keras.preprocessing.sequence.pad_sequences(X_paragraph_batch, maxlen=40, dtype='float32', padding='post')
        y_batch = np.array(y_batch)
        
        return (X_question_batch, X_paragraph_batch), y_batch

    def _on_epoch_end(self):
        random.shuffle(self.children)
        for article in self.children:
            random.shuffle(article.children)
        self._init_iter()

    def _get_next_datapoint(self):
        current_par = self.paragraph_list[self.PAR_LIST_IDX]
        X_question = self.active_question.data
        X_paragraph = current_par.data
        y = int((self.active_paragraph.id == current_par.id)
                and (self.active_question.is_impossible is False))
        return (X_question, X_paragraph, y)

    def _update_iter(self):
        self.PAR_LIST_IDX += 1
        if self.PAR_LIST_IDX >= len(self.paragraph_list):
            self.PAR_LIST_IDX = 0
            self._set_next_active_question()

