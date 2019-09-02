
import os
import zipfile
import urllib
import sys
import numpy as np
from keras import initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Embedding, Convolution1D, Dense, Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

# source : http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

shortWord_Dictionary = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

gloveFile = "./glove/glove.6B.300d.txt"

def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove():
    if not os.path.exists(gloveFile):
        glove_zip = 'glove/glove.6B.zip'

        if not os.path.exists('glove'):
            os.makedirs('glove')

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('glove')
        zip_ref.close()


def load_glove():
    download_glove()
    _word2em = {}
    file = open(gloveFile, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em

# defining model
    
def build_model(wordsCount,embeddingDimension,embeddingMatrix,wordLimit):
    model1 = Sequential()
    model1.add(Embedding(wordsCount, embeddingDimension, weights=[embeddingMatrix], input_length=wordLimit))
    model1.add(Dropout(0.1))
    model1.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model1.add(Dropout(0.1))
    # model1.add(Convolution1D(filters=filter, kernel_size=3, padding='same', activation='relu'))
    # model1.add(Dropout(0.3))
    model1.add(LSTM(256, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
                    dropout=0.1))
    model1.summary()

    model2 = Sequential()
    model2.add(Embedding(wordsCount, embeddingDimension, weights=[embeddingMatrix], input_length=wordLimit))
    model2.add(Dropout(0.1))
    model2.add(Convolution1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model2.add(Dropout(0.1))
    # model2.add(Convolution1D(filters=16, kernel_size=5, padding='same', activation='relu'))
    # model2.add(Dropout(0.3))
    model2.add(LSTM(256, activation=None, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
                    dropout=0.1))
    model2.summary()

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))
    model.add(Dense(256, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)))
    model.add(Dropout(0.1))
    # model.add(Dense(128 // 2, kernel_initializer=weights))
    # model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2), name='output'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, clipvalue=1.0))
    return model