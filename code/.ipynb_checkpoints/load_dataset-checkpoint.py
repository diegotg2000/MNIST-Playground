import numpy as np
import pandas as pd
import os




def load_dataframe(dataset):
    '''
    Arguments: dataset (str), either `font` or `mnist`.
    
    Outputs: a dataframe with the paths and labels of the images. The columns are [`path`, `label`]
    '''
    
    DATA_PATH = f'../data/{dataset}_data'


    paths = []
    digits = []
    for n in range(10):
        digit_path = os.path.join(DATA_PATH, f'{n}')
        images_names = os.listdir(digit_path)
        for im_name in images_names:
            im_path = os.path.join(digit_path, im_name)
            paths.append(im_path)
            digits.append(n)


    dataframe = pd.DataFrame({'path':paths, 'label':digits})

    return dataframe


def give_dataframe():
    
    mnist_df = load_dataframe('mnist')

    font_df = load_dataframe('font')

    df = pd.concat([mnist_df, font_df], ignore_index=True).sample(frac=1, random_state=51).reset_index(drop=True)

    return df


