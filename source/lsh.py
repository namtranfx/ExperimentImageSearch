
import numpy as np
import random
from collections import defaultdict
from itertools import combinations
from operator import itemgetter

classes_name = ['another', 'kitchen', 'dining_room', 'document', 'bathroom', 'bedroom', 'living_room', 'exterior']

def evaluation_metrics(y_preds, test_labels, num_neighbor):
    a = []
    true_labels = [x[1] for x in test_labels]
    for n in range(0, num_neighbor):
        pred_labels = []
        for pred in y_preds:
            top = min(n+1, len(pred))
            if top == 0:
                pred_labels.append(random.choice(classes_name))
            else:
                x = pred[:top]
                pred_labels.append(max(x, key = x.count))
        similars = [int(x == y) for x, y in zip(true_labels, pred_labels)]
        acc = sum(similars) / len(similars)
        a.append(acc)
    return a

def cosine_distance(u, v):
    return 1 - np.dot(u, v) / (np.dot(u, u)*np.dot(v, v))**0.5

class RandomProjection:
    
    def __init__(self, num_vecs, num_bits):
        self.randproj = [[random.gauss(0, 1) for __ in range(num_bits)] for _ in range(num_vecs)]
        
    def hash_func(self, point):
        prod = np.dot(point, self.randproj)
        return [(bit >= 0).astype(int) for bit in prod]

class CosineHashFamily:
    
    def __init__(self, num_bits):
        self.num_bits = num_bits

    def combine(self, hashcode):
        # return int(sum(bit * 2**i for i, bit in enumerate(hashcode)))
        return sum(bit * 2**i for i, bit in enumerate(hashcode)).astype(int)

class LSH:
    
    def __init__(self, hash_family, num_features):
        self.hash_family = hash_family
        self.num_features = num_features
        self.randproj = RandomProjection(num_features, self.hash_family.num_bits)
        self.hash_table = defaultdict(list)
        self.points = []

    def binary_hash(self, point): # return binary value
        # return self.randproj.hash_func(point[2])
        return self.randproj.hash_func(point)

    def hash_code(self, code): # return decimal value
        return self.hash_family.combine(code)

    def hash(self, point): # return decimal value
        return self.hash_code(self.binary_hash(point))

    def index(self, points): # create a hash table for dataset
        self.points = points
        for ix, point in enumerate(self.points):
            self.hash_table[self.hash(point)].extend([ix])

    def index_t(self, point, extra_data: int): # create a hash table for single element
        self.points.append(point)
        self.hash_table[self.hash(point)].extend([extra_data])

    def query(self, q, metric, num_neighbors):
        q_bits = self.binary_hash(q)
        candidates = set()
        num_diff_bit = 0
        while len(candidates) < num_neighbors and num_diff_bit < self.hash_family.num_bits/2:

            for diff_bits in combinations(range(self.hash_family.num_bits), num_diff_bit):
                alternate_bits = q_bits.copy()
                for i in diff_bits:
                    alternate_bits[i] = 1 if alternate_bits[i] == 0 else 0

                nearby_cluster = self.hash_code(alternate_bits)
                matches = self.hash_table.get(nearby_cluster, [])
                candidates.update(matches)

            num_diff_bit += 1

        # # rerank candidates
        # rerank = [(ix, metric(q[2], self.points[ix][2])) for ix in candidates]
        rerank = [(ix, metric(q, self.points[ix])) for ix in candidates]
        rerank.sort(key=itemgetter(1))
        amount_required = min(num_neighbors, len(rerank))
        return rerank[:amount_required]

class LSHTester:
    
    def __init__(self, points, queries, num_neighbors):
        self.points = points
        self.queries = queries
        self.num_neighbors = num_neighbors

    def build_lsh(self, hash_family, num_features):
        self.lsh = LSH(hash_family, num_features)
        self.lsh.index(self.points)

    def predict(self, metric):
        preds = []
        for q in self.queries:
            candidates = [ix for ix, distance in self.lsh.query(q, metric, self.num_neighbors)]
            preds.append([self.points[ix][1] for ix in candidates])
        return preds
    
