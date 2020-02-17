import numpy as np
import pandas as pd
import json
import os
import re
import random
import tensorflow as tf


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

    def __init__(self, fname=None, batch_size=1200, neg_rate=1., pos_aug=0, noise=0., shuffle=True):
        assert int(pos_aug) > 0
        assert noise >= 0 and noise < 1
        assert neg_rate >= 0 and neg_rate < 1
        assert int(batch_size) > 0
        assert type(shuffle) == bool

        self.children = []
        self.batch_size = int(batch_size)
        self.data_noise = noise
        self.shuffle = shuffle
        self.aug_rate = int(pos_aug)
        self.neg_sample_rate = neg_rate
        self.end_epoch = False
        random.seed()

        if fname is not None:
            self.read_json(fname)

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, idx):
        return self._get_batch()

    def _init_iter(self):
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
        # TODO add noise
        return aug_par

    def _create_paragraph_list(self, par_list):
        par_list.remove(self.active_paragraph)
        num_false_par = int(len(par_list) * self.neg_sample_rate)
        random.shuffle(par_list)
        par_list = par_list[:num_false_par]
        par_list.append(self.active_paragraph)
        for _ in range(self.aug_rate):
            aug_par = self._create_augmented_pos_paragraph()
            par_list.append(aug_par)
        random.shuffle(par_list)
        return par_list

    def _set_next_active_article(self):
        if self.active_article == self.children[-1]:
            self.end_epoch = True
        self.active_article_idx += 1
        self.active_article = self.children[self.active_article_idx]

    def _set_next_active_paragraph(self):
        if self.active_paragraph == self.active_article[-1]:
            self._set_next_active_article()
            self.active_paragraph_idx = -1
            self.active_paragraph = None
            self._set_next_active_paragraph()
        else:
            self.active_paragraph_idx += 1
            self.active_paragraph = self.active_article[self.active_paragraph_idx]

    def _set_next_active_question(self):
        if self.active_question == self.active_paragraph[-1]:
            self._set_next_active_paragraph()
            self.active_question_idx = -1
            self.active_question = None
            self._set_next_active_question()
            self._set_paragraph_list()
        else:
            self.active_question_idx += 1
            self.active_question = self.active_paragraph[self.active_question_idx]

    def _set_paragraph_list(self):
        par_list = self.active_article.children.copy()
        self.paragraph_list = self._create_paragraph_list(par_list)

    def train_test_split(self, test_size=0.25):
        assert test_size > 0 and test_size < 1
        random.shuffle(self.children)
        train = DataTree(batch_size=self.batch_size, pos_aug=self.aug_rate,
                         noise=self.data_noise, shuffle=self.shuffle, neg_rate=self.neg_sample_rate)
        test = DataTree(batch_size=self.batch_size, pos_aug=self.aug_rate,
                        noise=self.data_noise, shuffle=self.shuffle, neg_rate=self.neg_sample_rate)
        test_len = int(test_size * len(self.children))
        train.children = self.children[test_len:]
        test.children = self.children[:test_len]
        return train, test

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

    def transform_data(self, transform_func):
        def transform_child_data(node, trans_func=transform_func):
            if node.data is not None:
                node.data = trans_func(node.data)
                if node.type == "question":
                    node.data = node.data[0]
            for child in node.children:
                transform_child_data(child)

        for child in self.children:
            transform_child_data(child)

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
        X_batch = []
        y_batch = []
        for _ in range(self.batch_size):
            X, y = self._get_next_datapoint()
            X_batch.append(X)
            y_batch.append(y)
            self._update_iter()
            if self.end_epoch:
                # TODO
                break
        return np.array(X_batch), np.array(y_batch)

    
    def on_epoch_end(self):
        random.shuffle(self.children)
        for article in self.children:
            random.shuffle(article.children)
        self._init_iter()


    def _get_next_datapoint(self):
        current_par = self.paragraph_list[self.PAR_LIST_IDX]
        # X = np.array([np.array(self.active_question.data),
        #               np.array(current_par.data)])
        X = np.array([self.active_question.data,
                      current_par.data])
        y = int((self.active_paragraph.id == current_par.id)
                and (self.active_question.is_impossible is False))
        return X, y

    def _update_iter(self):
        self.PAR_LIST_IDX += 1
        if self.PAR_LIST_IDX >= len(self.paragraph_list):
            self.PAR_LIST_IDX = 0
            self._set_next_active_question()


# class DataGenerator(tf.keras.utils.Sequence):


# def atoi(text):
#     return int(text) if text.isdigit() else text


# def natural_keys(text):
#     return [atoi(c) for c in re.split(r'(\d+)', text)]

# if __name__ == "__main__":
#     article_data = np.load(
#         "/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/article_data.npy", allow_pickle=True)
#     question_data = pd.read_csv(
#         "/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/questions.csv")

    # data_tree = DataTree()

    # for i, article in enumerate(article_data):
    #     article_node = Node(ID=i, data_type="article")
    #     for j, paragraph in enumerate(article):
    #         paragraph_node = Node(
    #             data=paragraph["sents"], parent=article_node, ID=j, data_type="paragraph")
    #         questions = question_data.loc[(question_data["article"] == i) & (
    #             question_data["paragraph"] == j)]
    #         for _, q in questions.iterrows():
    #             question_node = Node(data=q["question_text"], parent=paragraph_node,
    #                                  data_type="question", is_impossible=q["is_impossible"])
    #             paragraph_node.add_child_node(question_node)
    #         article_node.add_child_node(paragraph_node)
    #     data_tree.children.append(article_node)
