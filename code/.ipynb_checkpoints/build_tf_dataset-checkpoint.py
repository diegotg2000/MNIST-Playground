import numpy as np
import pandas as pd
import tensorflow as tf



def load_image(image_path):
    
    file_image = tf.io.read_file(image_path)
        
    tensor_image = tf.io.decode_png(file_image, channels=0)

    tensor_image = tf.image.resize(tensor_image, [28, 28])
    
    return tensor_image


def give_classification_dataset(df):

    dataset = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values)).map(lambda x, y: (load_image(x), y))
    
    return dataset


def give_contrastive_dataset(df):
    
    f = lambda x,y,z: (load_image(x), load_image(y), z)
    
    dataset = tf.data.Dataset.from_tensor_slices((df.path1.values, df.path2.values, df.label.values)).map(f)
    
    return dataset







    






        








