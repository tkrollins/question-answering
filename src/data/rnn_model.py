import tensorflow as tf
import numpy as np
from data_generator import DataGenerator
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument('--data', type=str, default="/Users/tkrollins/OneDrive/Courses/capstone/question-answering/data/interim/SQuAD-v1.1-train_embeddings.json")
args = parser.parse_args()

q_input = tf.keras.layers.Input(shape=(768), name="Question_Input")
p_input = tf.keras.layers.Input(shape=(40,768), name="Paragraph_Input")

q_att = tf.keras.layers.Reshape((1,768))(q_input)

q_att = tf.keras.layers.Masking(mask_value=100., name="Question_Masking")(q_att)
p = tf.keras.layers.Masking(mask_value=0., name="Paragraph_Masking")(p_input)

ATT_ENCODER = tf.keras.layers.Dense(768, activation='relu', input_shape=(768,), name="Attention_Encoder")

# paragraph_encoded = tf.keras.layers.TimeDistributed(ATT_ENCODER)(paragraph_input, mask=paragraph_mask)
# p = ATT_ENCODER(p)
# q = ATT_ENCODER(q)

# p = tf.keras.layers.Dropout(0.3)(p)
# q = tf.keras.layers.Dropout(0.3)(q)



p = tf.keras.layers.Attention(name="Query-Question_Value-Paragraphs")([q_att, p])
p = tf.keras.backend.squeeze(p, axis=1)

q = tf.keras.layers.Dense(300, activation='relu', name="Question_Hidden", kernel_regularizer=tf.keras.regularizers.l1_l2())(q_input)
p = tf.keras.layers.Dense(300, activation='relu', name="Paragraph_Hidden", kernel_regularizer=tf.keras.regularizers.l1_l2())(p)

p = tf.keras.layers.Dropout(0.5)(p)
q = tf.keras.layers.Dropout(0.5)(q)

q = tf.keras.layers.Lambda(lambda x: x * 0.5)(q)
p = tf.keras.layers.Lambda(lambda x: x * 0.5)(p)

# combined = tf.keras.layers.concatenate(
#     [q, p], axis=-1)

combined = tf.keras.layers.add([q, p])

pred = tf.keras.layers.Dense(1, activation='sigmoid', name="Output")(combined)

model = tf.keras.Model(inputs=[q_input, p_input], outputs=pred)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=[tf.keras.metrics.Precision()])

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)


# def data_gen(data):
#     for X, y in data:
#         yield X, y

# BATCH_SIZE = 1000

# data = DataGenerator(args.data,
#                      neg_rate=.15, batch_size=BATCH_SIZE, noise=0.05, pos_aug=3)

# train_data, val_data = data.train_test_split(test_size=0.2)
# print(len(data))
# print(len(train_data))
# print(len(val_data))

# ckpt = tf.keras.callbacks.ModelCheckpoint("/content/drive/My Drive/capstone/question-answering/data/saved_models/att_model", verbose=1, save_best_only=True)
# stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)

# class_weights = {1 : 5., 0 : 1.}


# model.fit(x=data_gen(train_data), epochs=50, verbose=1, class_weight=class_weights, validation_data=data_gen(
#     val_data), steps_per_epoch=len(train_data), validation_steps=len(val_data), callbacks=[ckpt, stopping])
