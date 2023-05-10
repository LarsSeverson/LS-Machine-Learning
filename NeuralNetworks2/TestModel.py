import TrainModel
from tensorflow import keras
import numpy as np

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in TrainModel.word_index:
            encoded.append(TrainModel.word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

def test_model():
    model = keras.models.load_model('model.h5')
    with open('tester.txt') as f:
        for line in f.readlines():
            nline = line.replace(',', '').replace('.', '').replace('"', '').replace('(', '').replace(')', '').replace(':', '').strip().split(' ')
            encode = review_encode(nline)
            encode = keras.preprocessing.sequence.pad_sequences([encode], value=TrainModel.word_index['<PAD>'], padding='post',
                                                                   maxlen=250)
            predict = model.predict(encode)
            print(line)
            print(encode)
            print(predict[0])