import numpy as np
import os
from sentence_transformers import SentenceTransformer
import argparse as ap
from data_generator import DataGenerator
import spacy
from spacy.symbols import VERB
from nltk.stem.snowball import SnowballStemmer
import string

class SentenceParser(object):
    def __init__(self):
        self.sp = spacy.load('en_core_web_md')
        self.st = SnowballStemmer('english')

    @staticmethod
    def _remove_punctuation(text):
        no_punct = "".join([c for c in text if c not in string.punctuation])
        return no_punct

    def _get_roots(self, sent):
        roots = list(set([self.st.stem(chunk.root.head.text) for chunk in sent.noun_chunks if chunk.root.head.pos == VERB]))
        return roots

    @staticmethod
    def _get_named_entities(sent):
        ents = [ent.text.lower() for ent in sent.ents]
        return ents

    def parse_sent(self, text):
        sent = self.sp(self._remove_punctuation(text))
        ents = self._get_named_entities(sent)
        roots = self._get_roots(sent)
        return (ents, roots)

parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/external/albert")
parser.add_argument('--data', type=str,
                    default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-train_text.json")
parser.add_argument('--output', type=str,
                    default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-train_sBERT_ent-root_embeddings.json")


args = parser.parse_args()

MODEL_DIR = args.model
DATA_PATH = args.data
OUTPUT_DIR = args.output
print(f'MODEL_DIR = {MODEL_DIR}')
print(f'MODEL_DIR = {DATA_PATH}')
print(f'OUTPUT_DIR = {OUTPUT_DIR}')


def transform_wrapper(sBERT_model, sent_parser):
    def sentence_feature_transform(data):
        data_dict = {}
        if type(data)==list:
            is_question = False
        else:
            data = [data]
            is_question = True

        sbert_embeddings = sBERT_model.encode(data)
        sbert_embeddings = [emb.tolist() for emb in sbert_embeddings]
        if is_question:
            sbert_embeddings = sbert_embeddings[0]
        data_dict['embedding'] = sbert_embeddings

        data_dict['roots'] = []
        data_dict['ents'] = []
        for sent in data:
            ents, roots = sent_parser.parse_sent(sent)
            data_dict['ents'].append(ents)
            data_dict['roots'].append(roots)
            if is_question:
                data_dict['ents'] = np.array(data_dict['ents']).flatten().tolist()
                data_dict['roots'] = np.array(data_dict['roots']).flatten().tolist()
        return data_dict


    return sentence_feature_transform



if __name__ == '__main__':

    sbert_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    sent_parser = SentenceParser()


    sent_transform = transform_wrapper(sbert_model, sent_parser)

    data = DataGenerator(DATA_PATH)
    # data.write_json("./SQuAD-v1.1-train_text.json")
    data.transform_data(sent_transform)
    data.write_json(OUTPUT_DIR)
