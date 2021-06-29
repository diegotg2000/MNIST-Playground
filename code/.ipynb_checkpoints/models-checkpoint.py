import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseCategoricalCrossentropy, Accuracy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tqdm import tqdm


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

def give_cnn_block(kernel_size=3, filters=64, n=0):
    
    model = M.Sequential([L.Conv2D(filters, kernel_size), L.BatchNormalization(), L.ReLU()], name=f'cnn_block_{n}')
    
    return model


def give_ffl():    
    
    model = M.Sequential([L.Dense(64, activation='relu'), L.Dropout(0.2), L.Dense(10)], name='feed_forward')  
    
    return model


class FeatureExtractorCNN(M.Model):
    def __init__(self, kernels, filters):
        
        assert len(kernels) == len(filters), "Kernels and filters must have the same size"
        
        super(FeatureExtractorCNN, self).__init__()
        
        self.cnn_blocks = [give_cnn_block(k, f, i) for i, (k, f) in enumerate(zip(kernels, filters))]
        
        self.gba = L.GlobalAveragePooling2D()
        
    
    def call(self, inputs, training):
        x = inputs
        for i, cnn in enumerate(self.cnn_blocks):
            x = cnn(x, training)
        
        x = self.gba(x)
        
        return x        
        
        

class ClassCNN(M.Model):
    
    def __init__(self, kernels, filters):
        
        super(ClassCNN, self).__init__()
        
        self.feature_extractor = FeatureExtractorCNN(kernels, filters)
        
        self.head = give_ffl()
        
    def call(self, inputs, training=False):
        x = inputs/255.
        
        x = self.feature_extractor(x, training)
        
        return self.head(x, training)
            
    def fit(self, train_ds, val_ds, optimizer, epochs=5):
        
        accuracy = SparseCategoricalAccuracy()
        crossentropy = SparseCategoricalCrossentropy(from_logits=True)

        for epoch in range(epochs):

            print(f'Training on epoch {epoch+1}:')

            for batch, (images, labels) in tqdm(enumerate(train_ds)):

                with tf.GradientTape() as tape:

                    preds = self(images, training=True)

                    loss = sparse_categorical_crossentropy(labels, preds, from_logits=True)

                grads = tape.gradient(loss, self.trainable_weights)
                

                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                accuracy.update_state(labels, preds)

                crossentropy.update_state(labels, preds)

            print(f'TrainLoss: {crossentropy.result()}, TrainAcc: {accuracy.result()}')
            accuracy.reset_states()
            crossentropy.reset_states()

            for batch, (images, labels) in tqdm(enumerate(val_ds)):

                preds = self(images)

                accuracy.update_state(labels, preds)

                crossentropy.update_state(labels, preds)

            print(f'ValLoss: {crossentropy.result()}, ValAcc: {accuracy.result()}')
            accuracy.reset_states()
            crossentropy.reset_states()  
            

def contrastive_loss_fn(y_true, preds, margin=1):
    
    y = tf.cast(y_true, preds.dtype)
    
    return 0.5*y*preds**2 + 0.5*(1-y)*tf.math.square(tf.math.maximum(margin-preds, 0))

            
class ContrastiveCNN(M.Model):
    
    def __init__(self, emb_dim = 64, margin=1.0, extractor=None, kernels=None, filters=None):
        
        super(ContrastiveCNN, self).__init__()
        
        if extractor is not None:
        
            self.feature_extractor = extractor
            
        elif (kernels is not None) and (filters is not None):
            
            self.feature_extractor = FeatureExtractorCNN(kernels, filters)
            
        else: 
            raise Exception("Neither extractor nor kernels and filters passed")
        
        self.head = L.Dense(emb_dim, use_bias=False)
        
        self.margin = margin
        
    def call(self, inputs, training=False):
        x = inputs/255.
        
        x = self.feature_extractor(x, training)
        
        return self.head(x)
            
    def fit(self, train_ds, val_ds, optimizer, epochs=5):
        
        accuracy = Accuracy()

        for epoch in range(epochs):

            print(f'Training on epoch {epoch+1}:')

            for batch, (image1, image2, labels) in tqdm(enumerate(train_ds)):

                with tf.GradientTape() as tape:

                    emb1, emb2 = self(image1, training=True), self(image2, training=True)
                    
                    preds = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(emb1-emb2), axis=1))

                    loss = contrastive_loss_fn(labels, preds, margin=self.margin)

                grads = tape.gradient(loss, self.trainable_weights)
                
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                accuracy.update_state(labels, tf.where(preds<self.margin, 1, 0))


            print(f'TrainAcc: {accuracy.result()}')
            accuracy.reset_states()

            for batch, (image1, image2, labels) in tqdm(enumerate(val_ds)):

                emb1, emb2 = self(image1), self(image2)

                preds = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(emb1-emb2), axis=1))

                accuracy.update_state(labels, tf.where(preds<self.margin, 1, 0))


            print(f'ValAcc: {accuracy.result()}')
            accuracy.reset_states()

    
# x = tf.random.normal(shape=(32,28,28,1))    
# model = MyCNN([3,3,3,3,3,3,3], [16,32,32,32,64,64,128])
# y = model(x)

# print(model.summary())
