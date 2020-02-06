import tensorflow as tf
from tokenization import FullTokenizer
import tensorflow_hub as hub
import numpy as np

max_seq_length = 128  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")

bert_layer = hub.KerasLayer("/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/external/uncased_bert",
                            trainable=False)

# dense_layer = tf.keras.layers.Dense(16, activation='softmax')([input_word_ids, input_mask, segment_ids])

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

tokens_a = tokenizer.tokenize("The best name is Shakira.")
if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0 : (max_seq_length - 2)]

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

# print(tokens)
inputs = [np.array([input_ids]).astype(np.int32), np.array([input_mask]).astype(np.int32), np.array([segment_ids]).astype(np.int32)]
# print(inputs)
cls_embedding, _ = model.predict(x=inputs)
print(cls_embedding)