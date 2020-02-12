import numpy as np
import pandas as pd
import json


class Node(object):
    def __init__(self, data=None, parent=None, children=[], ID=None, data_type=None, is_impossible=None):
        self.id = ID
        self.data = data
        self.type = data_type
        self.parent = parent
        self.children = [child for child in children]
        self.is_impossible = is_impossible


class DataTree(object):

    def __init__(self, fname=None):
        self.children = []

        if fname is not None:
            self.read_json(fname)

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


# if __name__ == "__main__":
#     article_data = np.load(
#         "/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/article_data.npy", allow_pickle=True)
#     question_data = pd.read_csv(
#         "/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/questions.csv")

#     data_tree = DataTree()

#     for i, article in enumerate(article_data):
#         article_node = Node(ID=i, data_type="article")
#         for j, paragraph in enumerate(article):
#             paragraph_node = Node(
#                 data=paragraph["sents"], parent=article_node, ID=j, data_type="paragraph")
#             questions = question_data.loc[(question_data["article"] == i) & (
#                 question_data["paragraph"] == j)]
#             for _, q in questions.iterrows():
#                 question_node = Node(data=q["question_text"], parent=paragraph_node,
#                                      data_type="question", is_impossible=q["is_impossible"])
#                 paragraph_node.add_child_node(question_node)
#             article_node.add_child_node(paragraph_node)
#         data_tree.children.append(article_node)

