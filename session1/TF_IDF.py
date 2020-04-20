import os
from nltk.stem.porter import PorterStemmer


def listGroupData():
    trainDir, testDir = './20news-bydate/20news-bydate-train/', './20news-bydate/20news-bydate-test/'
    print(trainDir)
    listGroup = [group for group in os.listdir(trainDir)]
    listGroup.sort()
    with open("./english", "r") as f:
        stop_words = f.read().splitlines()
    stemmer = PorterStemmer()

    def colectDataFrom(parentDir, newsgroupList):
        data = []
        for groupId, newsgroup in enumerate(newsgroupList):
            label = groupId
            dirPath = parentDir + "/" + newsgroup + "/"
            files = [(filename, dirPath + filename)
                     for filename in os.listdir(dirPath) if os.path.isfile(dirPath + filename)]
            files.sort()
            for filename, filePath in files:
                with open(filePath, "r") as file:
                    text = file.read().lower()
                    text = re.sub(r'[_+]', '', text)
                    words = [stemmer.stem(word)
                             for word in re.split('\W+', text)
                             if word not in stop_words]

                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' +
                                filename + '<fff>' + content)
        return data

    trainData = colectDataFrom(parentDir=trainDir, newsgroupList=listGroup)
    testData = colectDataFrom(parentDir=testDir, newsgroupList=listGroup)
    fullData = trainData + testData
    with open('./20news-bydate/20news-train-processed.txt', 'w') as f:
        f.write('\n'.join(trainData))
    with open('./20news-test-processed.txt', 'w') as f:
        f.write('\n'.join(testData))
    with open('./20news-bydate/20news-full-processed.txt', 'w') as f:
        f.write('\n'.join(fullData))


def generateVocabulary(dataPath):
    def computeIdf(df, corpusSize):
        assert df > 0
        return np.log10(corpusSize * 1. / df)

    with open(dataPath) as f:
        lines = f.readlines()
    docCount = defaultdict(int)
    corpusSize = len(lines)
    for line in lines:
        features = line.split("<fff>")
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            docCount[word] += 1

    wordsIdfs = [(word, computeIdf(documentFreq, corpusSize))
                 for word, documentFreq in docCount.items()
                 if documentFreq > 10 and not word.isdigit()]
    wordsIdfs.sort(key=lambda wordIdf: wordIdf[0])
    wordsIdfs.sort(key=lambda wordIdf: -wordIdf[1])
    print(f'Vocabulary size: ', len(wordsIdfs))
    with open('./20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf)
                           for word, idf in wordsIdfs]))


def get_tf_idf(data_path):
    with open('./20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                      for line in f.readlines()]

        word_IDs = dict([(word, index)
                         for index, (word, idfs) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    with open(data_path) as f:
        documents = [
            (int(line.split('<fff>')[0]),
             int(line.split('<fff>')[1]),
             line.split('<fff>')[2])
            for line in f.readlines()
        ]

    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tfidfs = []
        sum_squares = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq / max_term_freq * idfs[word]
            words_tfidfs.append((word_IDs[word], tf_idf_value))
            sum_squares += tf_idf_value ** 2

        words_tfidfs_normalized = [str(index) + ':'
                                   + str(tf_idf_value /
                                         np.sqrt(sum_squares))
                                   for index, tf_idf_value in words_tfidfs]
        sparse_rep = ' '.join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    with open('./20news-bydate/data_tf_idf.txt', 'w') as f:
        f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep
                           for label, doc_id, sparse_rep in data_tf_idf]))
if __name__ == '__main__':
    listGroupData()
    generateVocabulary('datasets/20news-bydate/20news-full-processed.txt')
    get_tf_idf('datasets/20news-bydate/20news-train-processed.txt')
