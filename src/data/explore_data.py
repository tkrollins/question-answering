import json
import numpy as np

with open("../../data/raw/train-v2.0.json") as data:
    squad = json.load(data)['data']

sent_data = np.load("../../data/interim/article_data.npy", allow_pickle=True)

paragraphs = []
questions = []
contexts = []

for article in squad:
    paragraphs.append(len(article['paragraphs']))
    for paragraph in article['paragraphs']:
        questions.append(len(paragraph['qas']))
        contexts.append(len(paragraph['context']))

paragraphs = np.array(paragraphs)
questions = np.array(questions)
contexts = np.array(contexts)

sentences = []

for article in sent_data:
    for paragraph in article:
        sentences.append(len(paragraph['sents']))

print('Paragraphs Per Article')
print(f'MAX: {np.max(paragraphs)}')
print(f'MIN: {np.min(paragraphs)}')
print(f'MEAN: {np.mean(paragraphs)}')
print(f'STD: {np.std(paragraphs)}\n')

print('Questions Per Paragraph')
print(f'MAX: {np.max(questions)}')
print(f'MIN: {np.min(questions)}')
print(f'MEAN: {np.mean(questions)}')
print(f'STD: {np.std(questions)}\n')

print('Paragraph Character Length')
print(f'MAX: {np.max(contexts)}')
print(f'MIN: {np.min(contexts)}')
print(f'MEAN: {np.mean(contexts)}')
print(f'STD: {np.std(contexts)}\n')

print('Sentences Per Paragraph')
print(f'MAX: {np.max(sentences)}')
print(f'MIN: {np.min(sentences)}')
print(f'MEAN: {np.mean(sentences)}')
print(f'STD: {np.std(sentences)}')
