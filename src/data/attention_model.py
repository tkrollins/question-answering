import tensorflow as tf
import numpy as np
from data_generator import DataGenerator
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument('--data', type=str, default="data/interim/SQuAD-v1.1-train_embeddings.json")
args = parser.parse_args()

q_input = tf.keras.layers.Input(shape=(768), name="Question_Input")
p_input = tf.keras.layers.Input(shape=(40,768), name="Paragraph_Input")

q = tf.keras.backend.l2_normalize(q_input, axis=1)
p = tf.keras.backend.l2_normalize(p_input, axis=2)

q = tf.keras.layers.Reshape((1,768))(q)
q = tf.keras.layers.Masking(mask_value=100., name="Question_Masking")(q)
p = tf.keras.layers.Masking(mask_value=0., name="Paragraph_Masking")(p)

ATT_ENCODER = tf.keras.layers.Dense(768, activation='relu', input_shape=(768,), name="Attention_Encoder")

q = ATT_ENCODER(q)
p = ATT_ENCODER(p)

q = tf.keras.layers.Dropout(0.3)(q)
p = tf.keras.layers.Dropout(0.3)(p)

p = tf.keras.layers.Attention(name="Query-Question_Value-Paragraphs")([q, p])

q = tf.keras.backend.squeeze(q, axis=1)
p = tf.keras.backend.squeeze(p, axis=1)

pred = tf.keras.layers.dot([q, p], axes=1, normalize=True)


model = tf.keras.Model(inputs=[q_input, p_input], outputs=pred)

THRES = 0.5
METRICS = [
      tf.keras.metrics.Precision(name='precision', thresholds=THRES),
      tf.keras.metrics.Recall(name='recall', thresholds=THRES),
      tf.keras.metrics.TruePositives(name='tp', thresholds=THRES),
      tf.keras.metrics.FalsePositives(name='fp', thresholds=THRES),
      tf.keras.metrics.TrueNegatives(name='tn', thresholds=THRES),
      tf.keras.metrics.FalseNegatives(name='fn', thresholds=THRES), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=THRES),
]

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=METRICS)

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)


def data_gen(data):
    for X, y, weights in data:
        yield X, y, weights

BATCH_SIZE = 500

data = DataGenerator(args.data,
                     neg_rate=0.5, batch_size=BATCH_SIZE, noise=0.03, pos_aug=3, pos_weight=3)

train_data, val_data = data.train_test_split(test_size=0.2)
print(len(data))
print(len(train_data))
print(len(val_data))

ckpt = tf.keras.callbacks.ModelCheckpoint("data/saved_models/att_model_1", verbose=1, save_best_only=True)
stopping = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)


model.fit(x=data_gen(train_data), epochs=50, verbose=1, validation_data=data_gen(
    val_data), steps_per_epoch=len(train_data), validation_steps=len(val_data), callbacks=[ckpt, stopping])
