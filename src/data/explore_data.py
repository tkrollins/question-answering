import json
import numpy as np

with open("../../data/raw/train-v2.0.json") as data:
    squad = json.load(data)['data']

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
print(f'STD: {np.std(contexts)}')
