import tensorflow as tf
import numpy as np
from data_generator import DataGenerator
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument('--data', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-train_embeddings.json")
args = parser.parse_args()

question_input = tf.keras.layers.Input(shape=(768), name="Question_Input")
paragraph_input = tf.keras.layers.Input(shape=(40,768), name="Paragraph_Input")

question_masked = tf.keras.layers.Masking(mask_value=100., name="Question_Masking")(question_input)
paragraph_masked = tf.keras.layers.Masking(mask_value=0., name="Paragraph_Masking")(paragraph_input)

ATT_ENCODER = tf.keras.layers.Dense(768, activation='relu', input_shape=(768,), name="Attention_Encoder")

# paragraph_encoded = tf.keras.layers.TimeDistributed(ATT_ENCODER)(paragraph_input, mask=paragraph_mask)
paragraph_encoded = ATT_ENCODER(paragraph_masked)
question_encoded = ATT_ENCODER(question_masked)

paragraph_attention = tf.keras.layers.Attention(name="Query-Question_Value-Paragraphs")([question_encoded, paragraph_encoded])

question_emb = tf.keras.layers.Dense(300, activation='relu', name="Question_Hidden")(question_encoded)
paragraph_emb = tf.keras.layers.Dense(300, activation='relu', name="Paragraph_Hidden")(paragraph_attention[:,0,:])

combined_emb = tf.keras.layers.concatenate(
    [question_emb, paragraph_emb], axis=-1)

pred = tf.keras.layers.Dense(1, activation='sigmoid', name="Output")(combined_emb)

model = tf.keras.Model(inputs=[question_input, paragraph_input], outputs=pred)

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=[tf.keras.metrics.Precision()])

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)
exit(0)

def data_gen(data):
    for X, y in data:
        yield X, y

BATCH_SIZE = 1000

data = DataGenerator(args.data,
                     neg_rate=.15, batch_size=BATCH_SIZE, noise=0., pos_aug=3)

train_data, val_data = data.train_test_split(test_size=0.2)
print(len(data))
print(len(train_data))
print(len(val_data))

ckpt = tf.keras.callbacks.ModelCheckpoint("/content/drive/My Drive/capstone/question-answering/data/saved_models/att_model", verbose=1, save_best_only=True)
stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)

class_weights = {1 : 20., 0 : 1.}


model.fit(x=data_gen(train_data), epochs=1, verbose=1, class_weight=class_weights, validation_data=data_gen(
    val_data), steps_per_epoch=len(train_data), validation_steps=len(val_data), callbacks=[ckpt, stopping])
