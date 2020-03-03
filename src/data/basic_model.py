import tensorflow as tf
import numpy as np
from data_generator import DataGenerator


def data_gen(data):
    for X, y in data:
        yield X, y


question_input = tf.keras.layers.Input(shape=(768))
paragraph_input = tf.keras.layers.Input(shape=(768))

question_emb = tf.keras.layers.Dense(300, activation='relu')(question_input)
paragraph_emb = tf.keras.layers.Dense(300, activation='relu')(paragraph_input)

combined_emb = tf.keras.layers.concatenate(
    [question_emb, paragraph_emb], axis=-1)

pred = tf.keras.layers.Dense(1, activation='sigmoid')(combined_emb)

model = tf.keras.Model(inputs=[question_input, paragraph_input], outputs=pred)

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

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=METRICS)
model.summary()

BATCH_SIZE = 1000

data = DataGenerator("data/interim/SQuAD-v1.1-train_embeddings.json",
                     neg_rate=.5, batch_size=BATCH_SIZE)

train_data, val_data = data.train_test_split(test_size=0.2)
print(len(data))
print(len(train_data))
print(len(val_data))

ckpt = tf.keras.callbacks.ModelCheckpoint("data/saved_models/avg_model.ckpt", verbose=1, save_best_only=True)
stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)


model.fit(x=data_gen(train_data), epochs=1, verbose=1, validation_data=data_gen(
    val_data), steps_per_epoch=len(train_data), validation_steps=len(val_data), callbacks=[ckpt, stopping])
