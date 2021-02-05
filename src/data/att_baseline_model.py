import tensorflow as tf
import numpy as np
from data_generator import DataGenerator
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument('--train_data', type=str, default="data/interim/SQuAD-v1.1-train_sBERT_embeddings-T.json")
parser.add_argument('--val_data', type=str, default="data/interim/SQuAD-v1.1-train_sBERT_embeddings-D.json")
parser.add_argument('--checkpoint', type=str, default="data/saved_models/base_att_model_1")
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--pos_noise', type=float, default=0.03)
parser.add_argument('--learn_rate', type=float, default=0.001)
parser.add_argument('--neg_samples', type=float, default=0.5)
parser.add_argument('--pos_aug_rate', type=int, default=3)
parser.add_argument('--pos_sample_weight', type=int, default=3)
parser.add_argument('--early_stopping', type=int, default=5)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

q_input = tf.keras.layers.Input(shape=(768), name="Question_Input")
p_input = tf.keras.layers.Input(shape=(40,768), name="Paragraph_Input")

q = tf.keras.backend.l2_normalize(q_input, axis=1)
p = tf.keras.backend.l2_normalize(p_input, axis=2)

q = tf.keras.layers.Reshape((1,768))(q)
q = tf.keras.layers.Masking(mask_value=100., name="Question_Masking")(q)
p = tf.keras.layers.Masking(mask_value=0., name="Paragraph_Masking")(p)

p = tf.keras.layers.Attention(name="Query-Question_Value-Paragraphs", scale=True)([q, p])

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
              optimizer=tf.keras.optimizers.Adam(learning_rate=args.learn_rate), metrics=METRICS)

model.summary()
model.save(args.checkpoint)

tf.keras.utils.plot_model(model, show_shapes=True)