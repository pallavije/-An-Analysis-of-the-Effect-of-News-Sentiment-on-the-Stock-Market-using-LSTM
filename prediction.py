
import re
import numpy as np
from nltk.corpus import stopwords
from util import shortWord_Dictionary, build_model, load_glove

stop = stopwords.words('english')

class preprocess:
    def __init__(self):
        self.word2int = np.load("./word_2_vector.npy", allow_pickle=True).item()
        self.wordLimit = 200
        self.embeddingDimension = 300

    def cleaning(self):
        shortword_free = [shortWord_Dictionary[word] if word in shortWord_Dictionary else word for word in
                          self.news.lower().split()]
        only_alphanumeric = re.sub("[^a-zA-Z0-9]", " ", ' '.join(shortword_free))
        stopword_free = [word for word in only_alphanumeric.split() if word not in stop]
        self.cleanedText = " ".join(stopword_free)

    def text2Vec(self):
        self.vector = list()
        self.cleaning()
        for words in self.cleanedText.split():
            if words in self.word2int:
                self.vector.append(self.word2int[words])
            else:
                self.vector.append(self.word2int['UNK'])

    def sequenceCheck(self, news):
        self.news = news
        self.text2Vec()
        self.newSequence = self.vector
        if len(self.newSequence) >= self.wordLimit:
            self.newSequence = self.newSequence[:self.wordLimit]
        else:
            for i in range(self.wordLimit - len(self.newSequence)):
                self.newSequence.append(self.word2int['PAD'])
        return self.newSequence

    def embedding(self):
        self.embeddingMatrix = np.zeros((len(self.word2int), self.embeddingDimension))
        self.word2emb = load_glove()
        for word, vec in self.word2int.items():
            if word in self.word2emb:
                self.embeddingMatrix[vec] = self.word2emb[word]
            else:
                tempEmbedding = np.array(np.random.uniform(-1.0, 1.0, self.embeddingDimension))
                self.word2emb[word] = tempEmbedding
                self.embeddingMatrix[vec] = tempEmbedding

    def modeling(self):
        self.embedding()
        model = build_model(len(self.word2int), 300, self.embeddingMatrix, self.wordLimit)
        model.load_weights("./model.h5")
        return model


def reverse_normalize(cost):
    misc = np.load("./misc.npy", allow_pickle=True).item()
    return cost[0][0] * (misc["maxCost"] - misc["minCost"]) + misc["minCost"]


'''

def prediction(news):
    model = preprocess().modeling()
    data = preprocess().sequenceCheck(news)
    preprocessedData = np.array(data).reshape((1, -1))
    predicted = model.predict([preprocessedData, preprocessedData])
    print(predicted)
    answer = reverse_normalize(predicted)
    print(answer)
    print("The Apple should open : {} from the previous open.".format(np.round(answer, 2)))


news = "Leaked document reveals Apple conducted research to target emotionally vulnerable and insecure youth"

# Woman says note from Chinese 'prisoner' was hidden in new purse. 21,000 AT&T workers poised for Monday strike housands march against Trump climate policies in D.C., across USA  Kentucky judge won't hear gay adoptions because it's not in the child's \"best interest\"Multiple victims shot in UTC area apartment complex Drones Lead Police to Illegal Dumping in Riverside County | NBC Southern California An 86-year-old Californian woman has died trying to fight a man who was allegedly sexually assaulting her 61-year-old friend."

prediction(news)

'''
