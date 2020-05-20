import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def load_data(pathFile, vocab_size):
    data = []
    label = []
    with open(pathFile) as f:
        list_line = f.read().split("\n")
        for line in list_line:
            features = line.split("<fff>")
            # get label and id of document
            label_, id_ = int(features[0]), int(features[1])
            # update label count of label
            embed = [0.0 for _ in range(vocab_size)]
            list_words = features[2].split(" ")
            for word in list_words:
                index = int(word.split(":")[0])
                tf_idf_word = float(word.split(":")[1])
                # performances tf idf of document
                embed[index] = tf_idf_word
            # add member
            data.append(np.array(embed))
            label.append(label_)
        return np.array(data), np.array(label)


def with_kmeans(pathFile, vocab_size):
    data, label = load_data(pathFile, vocab_size)
    kmeans = KMeans(n_clusters=20, init="random",
                    n_init=5, tol=1e-3, random_state=2018)
    kmeans.fit(data)
    return kmeans.labels_


def with_linear_SVMS(pathFile, vocab_size):
    x, y = load_data(pathFile, vocab_size)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    classifier = LinearSVC(C=10.0, tol=0.001, verbose=True)
    classifier.fit(X_train, y_train)
    predict_y = classifier.predict(X_test)
    acc = accuracy(predict_y, y_test)
    return acc


def accuracy(label_predict, label_expected):
    match = np.equal(label_predict, label_expected)
    acc = np.sum(match.astype(float)) / label_expected.size
    return acc

if __name__ == "__main__":
	with_kmeans("./data_tf_idf.txt", 14140)
	acc_linear = with_linear_SVMS("./data_tf_idf.txt", 14140)
    print("accuracy linear SVC ",acc_linear)
