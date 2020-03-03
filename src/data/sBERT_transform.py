import numpy as np
import os
from sentence_transformers import SentenceTransformer
import argparse as ap
from data_generator import DataGenerator

parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/external/albert")
parser.add_argument('--data', type=str,
                    default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/raw/SQuAD-v1.1-train.json")
parser.add_argument('--output', type=str,
                    default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-train_sBERT_embeddings.json")


args = parser.parse_args()

MODEL_DIR = args.model
DATA_PATH = args.data
OUTPUT_DIR = args.output
print(f'MODEL_DIR = {MODEL_DIR}')
print(f'MODEL_DIR = {DATA_PATH}')
print(f'OUTPUT_DIR = {OUTPUT_DIR}')


def sbert_transform_wrapper(model):
    def sbert_transform(data):
        data = [data] if type(data)!=list else data
        sbert_embeddings = model.encode(data)
        sbert_embeddings = [emb.tolist() for emb in sbert_embeddings]
        return sbert_embeddings

    return sbert_transform



if __name__ == '__main__':

    sbert_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')


    sbert_transform = sbert_transform_wrapper(sbert_model)

    data = DataGenerator(DATA_PATH)
    # data.write_json("./SQuAD-v1.1-train_text.json")
    data.transform_data(sbert_transform)
    data.write_json(OUTPUT_DIR)
