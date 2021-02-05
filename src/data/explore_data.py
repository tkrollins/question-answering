import json
import numpy as np
from data_generator import DataGenerator


data = DataGenerator("/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-dev_embeddings.json")

paragraphs = []
questions = []
# contexts = []
sentences = []

for article in data.children:
    paragraphs.append(len(article))
    for paragraph in article:
        sentences.append(len(paragraph.data))
        questions.append(len(paragraph))

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

# print('Paragraph Character Length')
# print(f'MAX: {np.max(contexts)}')
# print(f'MIN: {np.min(contexts)}')
# print(f'MEAN: {np.mean(contexts)}')
# print(f'STD: {np.std(contexts)}\n')

print('Sentences Per Paragraph')
print(f'MAX: {np.max(sentences)}')
print(f'MIN: {np.min(sentences)}')
print(f'MEAN: {np.mean(sentences)}')
print(f'STD: {np.std(sentences)}')
