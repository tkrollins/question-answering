import tensorflow as tf
from data_generator import DataGenerator, Node
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument('--data', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-dev_sBERT_embeddings.json")
parser.add_argument('--model', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/saved_models/att_model-best")
parser.add_argument('--output', type=str, default="results.csv")
parser.add_argument('--top_n', type=int, default=5)
args = parser.parse_args()



class ParagraphPredictor(object):

    def __init__(self, model, articles):
        self.model = model
        self.articles = articles if type(articles) == list else [articles]

    def ask_question(self):
        for article in self.articles:
            article_data = np.array([par.data for par in article])
            article_data = tf.keras.preprocessing.sequence.pad_sequences(
                article_data, maxlen=40, dtype='float32', padding='post')
            for p_id, paragraph in enumerate(article):
                for question in paragraph:
                    yield (np.array([question.data] * len(article_data)), article_data, article.id, p_id)

    @staticmethod
    def _get_results_dict():
        results = defaultdict(list)
        results["top_par_pred_score"] = []
        results["answer_par_pred_score"] = []
        results["article_id"] = []
        results["pred_paragraph"] = []
        results["answer_paragraph"] = []
        results["correct"] = []
        results["in_top_2"] = []
        results["in_top_3"] = []
        results["in_top_4"] = []
        results["in_top_5"] = []
        return results

    def get_paragraph_scores(self, fname, top_n):
        results = self._get_results_dict()
        prev_aid = self.articles[0].id
        article_count = 0
        for question, article, article_id, answer in self.ask_question():
            if article_id != prev_aid:
                assert len(results) == 10
                length = len(results["correct"])
                for key in results:
                    assert len(results[key]) == length
                results_df = pd.DataFrame(results)
                assert results_df.shape[1] == 10
                assert results_df.shape[0] == length
                print("Writing to CSV...")
                if not os.path.isfile(fname):
                    print(f"Creating  {fname}...")
                    results_df.to_csv(fname, header='column_names')
                else: # else it exists so append without writing the header
                    print(f"Appending to {fname}...")
                    results_df.to_csv(fname, mode='a', header=False)
                article_count += 1
                print(f'Articles so far: {article_count}')
                results = self._get_results_dict()
                prev_aid = article_id
            
            pred_scores = self.model.predict((question, article)).flatten()
            pred_par = np.argmax(pred_scores)
            results["top_par_pred_score"].append(pred_scores[pred_par])
            results["answer_par_pred_score"].append(pred_scores[answer])

            temp = pred_par
            in_top_n = [(pred_par == answer)] * top_n
            i = 1
            while (not in_top_n[i-1]) and (i < top_n):
                pred_scores[temp] = 0
                temp = np.argmax(pred_scores)
                in_top_n[i:top_n] = [(temp == answer)] * (top_n - i) 
                i += 1
                
            results["article_id"].append(article_id)
            results["pred_paragraph"].append(pred_par)
            results["answer_paragraph"].append(answer)
            results["correct"].append((pred_par == answer))
            results["in_top_2"].append(in_top_n[1])
            results["in_top_3"].append(in_top_n[2])
            results["in_top_4"].append(in_top_n[3])
            results["in_top_5"].append(in_top_n[4])


        assert len(results) == 10
        length = len(results["correct"])
        for key in results:
            assert len(results[key]) == length
        results_df = pd.DataFrame(results)
        assert results_df.shape[1] == 10
        assert results_df.shape[0] == length
        print("Writing to CSV...")
        if not os.path.isfile(fname):
            print(f"Creating  {fname}...")
            results_df.to_csv(fname, header='column_names')
        else: # else it exists so append without writing the header
            print(f"Appending to {fname}...")
            results_df.to_csv(fname, mode='a', header=False)


val_data = DataGenerator(args.data)

model = tf.keras.models.load_model(args.model)

predictor = ParagraphPredictor(model, val_data.children)

predictor.get_paragraph_scores(args.output, args.top_n)