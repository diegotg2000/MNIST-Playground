import numpy as np
import PIL
import tensorflow as tf
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



DATA_PATH = '../data/mnist_data'


x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)


#print(x.shape, y.shape)




for image_array, label in zip(x, y):

    image = PIL.Image.fromarray(255-image_array, mode='L')

    path = os.path.join(DATA_PATH, f'{label}')
    
    #print(path)
    
    try: 
        prev = len(os.listdir(path))
        
    except: 
        os.system(f'mkdir {path}')
        prev = 0
        
    image_path = path + f'/image_{prev}.png'  
    image.save(image_path)
