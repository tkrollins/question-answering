import tensorflow as tf
import bert
import albert_tokenizer
import tensorflow_hub as hub
import numpy as np
import os
import sentencepiece as spm
import argparse as ap
from data_generator import DataGenerator

parser = ap.ArgumentParser(description='Extract ALBERT features')
parser.add_argument('--model', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/external/albert")
parser.add_argument('--data', type=str,
                    default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/raw/SQuAD-v1.1-train.json")
parser.add_argument('--output', type=str,
                    default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-train_embeddings.json")


args = parser.parse_args()

MODEL_DIR = args.model
DATA_PATH = args.data
OUTPUT_DIR = args.output
print(f'MODEL_DIR = {MODEL_DIR}')
print(f'MODEL_DIR = {DATA_PATH}')
print(f'OUTPUT_DIR = {OUTPUT_DIR}')


def create_bert(bert_path, max_seq_length=256):
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    bert_layer = hub.KerasLayer(
        bert_path, trainable=False, signature="tokens", output_key="pooled_output")
    bert_output = bert_layer(
        dict(input_ids=input_word_ids, input_mask=input_mask, segment_ids=segment_ids))
    bert_embedding_model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=bert_output)
    return bert_embedding_model


def create_bert_input_sentence(sentence, tokenizer, max_seq_length):
    clean_sentence = bert.albert_tokenization.preprocess_text(sentence)
    tokens_a = tokenizer.tokenize(clean_sentence)

    if len(tokens_a) > max_seq_length - 2:
        print("TOO BIG")
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (np.array(input_ids).astype(np.int32),
            np.array(input_mask).astype(np.int32),
            np.array(segment_ids).astype(np.int32))


def create_bert_input_paragraph(paragraph, tokenizer, max_seq_length):
    bert_paragraph_input_ids = []
    bert_paragraph_input_mask = []
    bert_paragraph_segment_ids = []

    for sent in paragraph:
        input_ids, input_mask, segment_ids = create_bert_input_sentence(
            sent, tokenizer, max_seq_length)
        bert_paragraph_input_ids.append(input_ids)
        bert_paragraph_input_mask.append(input_mask)
        bert_paragraph_segment_ids.append(segment_ids)

    return [bert_paragraph_input_ids,
            bert_paragraph_input_mask,
            bert_paragraph_segment_ids]


def get_bert_ouput_paragraph(bert_model, paragraph_input):
    paragraph_embeddings = bert_model.predict(paragraph_input)
    assert len(paragraph_embeddings) == len(paragraph_input[0])
    return paragraph_embeddings

def bert_transform_wrapper(tokenizer, model, max_seq_length=128):
    def bert_transform(data):
        data = [data] if type(data)!=list else data
        bert_input = create_bert_input_paragraph(data, tokenizer, max_seq_length)
        bert_embeddings = get_bert_ouput_paragraph(model, bert_input)
        return bert_embeddings.tolist()

    return bert_transform



if __name__ == '__main__':

    MAX_SEQ_LEN = 256

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    bert_model = create_bert(MODEL_DIR, max_seq_length=MAX_SEQ_LEN)
    bert_model.summary()

    vocab_file = os.path.join(MODEL_DIR, "assets", "30k-clean.vocab")
    spm_file = os.path.join(MODEL_DIR, "assets", "30k-clean.model")

    # tokenizer = create_tokenizer(bert_path)
    tokenizer = albert_tokenizer.FullTokenizer(
        vocab_file, spm_model_file=spm_file)

    bert_transform = bert_transform_wrapper(tokenizer, bert_model, max_seq_length=MAX_SEQ_LEN)

    data = DataGenerator(DATA_PATH, sqaud=True)
    data.write_json("./SQuAD-v1.1-train_text.json")
    data.transform_data(bert_transform)
    data.write_json(OUTPUT_DIR)
