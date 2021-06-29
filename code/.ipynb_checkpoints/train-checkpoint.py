import numpy as np
import pandas as pd
import tensorflow as tf
import build_tf_dataset
import load_dataset
import models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam


EPOCHS = 6
TEST_SIZE = 500
TRAIN_BATCH = 64
TEST_BATCH = 32
kernels, filters = [3,3,3,3,3,3,3], [16,32,32,32,64,64,128]

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


class_df = load_dataset.give_dataframe()#.sample(frac=0.01).reset_index(drop=True)

train_df, val_df = train_test_split(class_df, test_size=TEST_SIZE, random_state=0)

train_ds = build_tf_dataset.give_classification_dataset(train_df).batch(TRAIN_BATCH)

val_ds = build_tf_dataset.give_classification_dataset(val_df).batch(TEST_BATCH)




class_model = models.ClassCNN(kernels, filters)
contrastive_model = models.ContrastiveCNN(kernels=kernels, filters=filters)


inputs = tf.random.normal((1,28,28,1))
class_model(inputs)
contrastive_model(inputs)



optimizer = Adam()

print('Training Classification model')
class_model.fit(train_ds, val_ds, optimizer, epochs=EPOCHS)


contrastive_model.feature_extractor.set_weights(class_model.feature_extractor.get_weights())


contrastive_df = pd.read_csv('contrastive.csv')

train_df, val_df = train_test_split(contrastive_df, test_size=TEST_SIZE, random_state=0)

train_ds = build_tf_dataset.give_contrastive_dataset(train_df).batch(TRAIN_BATCH)

val_ds = build_tf_dataset.give_contrastive_dataset(val_df).batch(TEST_BATCH)

print('Training Contrastive model')

contrastive_model.fit(train_ds, val_ds, optimizer, epochs=EPOCHS)


class_model.save('../models/class_model')

contrastive_model.save('../models/contrastive_model')


print(contrastive_model.summary())





