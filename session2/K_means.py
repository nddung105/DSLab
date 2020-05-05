import os
import numpy as np
from collections import defaultdict
import argparse


class Member():
        # Class Member for one point data

    def __init__(self, embed_tf_idf, label=None, _id=None):
        self.label = label  # type is int from 0 to (number clusters-1)
        self.id = _id  # id of document. type is int
        self.embed_tf_idf = embed_tf_idf  # performances tf idf of document.
        # type is array. (1,n) n is vocab size


class Cluster():
        # Class cluster for one label

    def __init__(self):
        self.centroid = None  # type is array. is embed_tf_idf of member
        self.members = []  # type is Member. list members of cluster


class Kmeans():

    def __init__(self, number_clusters):
        self.number_clusters = number_clusters  # type int. Number of lable
        # list of clusters. length = number of label
        self.clusters = [Cluster() for _ in range(self.number_clusters)]
        self.E = []  # list of centroid
        self.S = 0  # overall similarity

    def load_data(self, path_data, vocab_size):
        self.data = []  # list of members
        # dictionary. Key is label, value is number member of label
        self.label_count = {i: 0 for i in range(20)}
        # get list line from data file
        with open(path_data) as f:
            list_line = f.read().split("\n")
        for line in list_line:
            features = line.split("<fff>")
            # get label and id of document
            label, id_ = int(features[0]), int(features[1])
            # update label count of label
            self.label_count[label] += 1
            embed = [0.0 for _ in range(vocab_size)]
            list_words = features[2].split(" ")
            for word in list_words:
                index = int(word.split(":")[0])
                tf_idf_word = float(word.split(":")[1])
                # performances tf idf of document
                embed[index] = tf_idf_word
            # add member
            self.data.append(Member(np.array(embed), label=label, _id=id_))
        print("number data: " + str(len(self.data)))
        print("-------load data success--------")

    def random_init(self, seed_value):
        index = []
        i = 0
        for member in self.data:
            if (member.label not in index):
                self.clusters[i].centroid = member.embed_tf_idf
                i += 1
                index.append(member.label)

    def compute_similarity(self, member, centroid):
        # calculate norm 2 of (member embed - centroid)
        euclid_dist = np.linalg.norm(
            member.embed_tf_idf - centroid)
        return euclid_dist

    def select_clusters(self, member):
        best_cluster = None
        min_similarity = 100000
        for cluster in self.clusters:
            similarity = self.compute_similarity(member, cluster.centroid)
            if similarity < min_similarity:
                best_cluster = cluster
                min_similarity = similarity
        # best cluster is cluster has norm (centroid-member) min
        best_cluster.members.append(member)  # append member to cluster
        return min_similarity

    def update_centroid(self, cluster):
        # compute new centroid to cluster after each epoch
        members_embed = [member.embed_tf_idf for member in cluster.members]
        aver_embed = np.mean(members_embed, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_embed**2))
        # include normal data
        new_centroid = np.array([value / sqrt_sum_sqr for value in aver_embed])
        cluster.centroid = new_centroid

    def compute_purity(self):
        majority_sum = 0
        for cluster in self.clusters:
            members_label = [member.label for member in cluster.members]
            max_count = max([members_label.count(label)
                             for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self.data)

    def compute_NMI(self):
        # normalized mutual information
        I_value, H_omega, H_C, N = 0., 0., 0., len(self.data)
        for cluster in self.clusters:
            wk = len(cluster.members) * 1.
            H_omega += -wk / N * np.log10(wk / N)
            members_label = [member.label for member in cluster.members]
            for label in range(20):
                wk_cj = members_label.count(label) * 1.
                cj = self.label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self.label_count[label] * 1.
            H_C += -cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)

    def stop_condition(self, criterion, threshold):
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if(criterion == "max_iters"):
            # stop with max iters. it is number of loop
            if (self.iteration >= threshold):
                return True
            else:
                return False
        elif (criterion == "centroid"):
            # stop with centroid if list centroid not change
            E_new = [list(cluster.centroid) for cluster in self.clusters]
            E_new_minus_E = [
                centroid for centroid in E_new if centroid not in self.E]
            self.E = E_new
            if(len(E_new_minus_E) <= threshold):
                return True
            else:
                return False
        else:
            # stop with similarity if loss < threshold
            new_S_minus_S = self.new_S - self.S
            if (new_S_minus_S < 0):
                new_S_minus_S = - new_S_minus_S
            self.S = self.new_S
            if(new_S_minus_S <= threshold):
                return True
            else:
                return False

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)
        self.iteration = 0
        while True:
            print("epoch " + str(self.iteration + 1))
            for cluster in self.clusters:
                cluster.members = []
            self.new_S = 0
            for member in self.data:
                min_s = self.select_clusters(member)
                self.new_S += min_s
            for cluster in self.clusters:
                self.update_centroid(cluster)
            score = self.acc()
            print("epoch " + str(self.iteration + 1) +
                  " complete. Acc " + str(score))
            print("purity: " + str(self.compute_purity()) +
                  " NMI: " + str(self.compute_NMI()))
            self.iteration += 1
            if(self.stop_condition(criterion, threshold)):
                break

    def acc(self):
        # accuracy = (point predict true) / (number of point predict)
        true = 0
        data_size = len(self.data)
        for indexI in range(self.number_clusters):
            size = len(self.clusters[indexI].members)
            for indexJ in range(size):
                if (self.clusters[indexI].members[indexJ].label == indexI):
                    true += 1
        return true / data_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Kmeans Cluster')
    # get path of data file
    parser.add_argument('path', metavar='--path', type=str, nargs='?',
                        help='the path of data file', default="./data_tf_idf.txt")
    # get vocab size
    parser.add_argument('vocab_size', metavar='--vocab_size',
                        type=int, help='size of vocab', default=14140, nargs='?')
    # get number of cluster. It is also number of label
    parser.add_argument('clusters', metavar='--clusters', type=int,
                        help='number of clusters', default=20, nargs='?')
    # get number of threshold
    parser.add_argument('threshold', metavar='--threshold',
                        type=int, help='number threshold', default=10, nargs='?')
    # get criterion
    parser.add_argument('criterion', metavar='--criterion', type=str,
                        help='include "centroid", "similarity", "max_iters"', default="max_iters", nargs='?')
    args = parser.parse_args()

    kmean = Kmeans(args.clusters)
    kmean.load_data(args.path, args.vocab_size)
    kmean.run(0, args.criterion, args.threshold)
