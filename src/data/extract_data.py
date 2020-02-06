import numpy as np
import pandas as pd
import json
from collections import defaultdict
import spacy


def sentence_segment(text):
    nlp = spacy.load("en_core_web_sm")
    nlp_text = nlp(text)
    return {'named_ents': [ent.string.strip() for ent in nlp_text.ents], 'sents': [sent.text for sent in nlp_text.sents]}


def main():
    with open("/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/raw/train-v2.0.json") as data:
        squad = json.load(data)['data']

    articles = []
    questions = defaultdict(list)

    for i, article in enumerate(squad):
        paragraphs = []
        for j, paragraph in enumerate(article['paragraphs']):
            for Q in paragraph['qas']:
                questions['question_text'].append(Q['question'])
                questions['is_impossible'].append(Q['is_impossible'])
                questions['article'].append(i)
                questions['paragraph'].append(j)
            paragraphs.append(sentence_segment(paragraph['context']))
        articles.append(np.array(paragraphs))
        print(f'Article {i}')

    articles = np.array(articles)
    print(articles.shape)
    np.save('../../data/interim/article_data.npy', articles)
    questions = pd.DataFrame(questions)
    questions.to_csv("../../data/interim/questions.csv", index=False)

if __name__ == '__main__':
    main()