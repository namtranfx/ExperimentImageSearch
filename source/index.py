import faiss
from faiss import write_index, read_index
from annoy import AnnoyIndex


class MyIndex:
    def __init__(self) -> None:
        pass
    def saveIndex(self, path):
        pass
    def loadIndex(self, path):
        pass
    def add(self, feature_vec):
        pass
    # Return an array of image indices
    def search(self, query_feature_vec, k_top):
        pass

class FaissRawIndex(MyIndex):
    def __init__(self, feature_size) -> None:
        super().__init__()
        self.feature_size = feature_size
        self._faiss_index = faiss.IndexFlatL2(feature_size)   # build the index
        self.is_finished = False
    def saveIndex(self, path):
        write_index(self._faiss_index, path)
        self.is_finished = True
    def loadIndex(self, path):
        self._faiss_index = read_index(path)
    def add(self, feature_vec):
        if self.is_finished == True:
            self._faiss_index.reset()
            self.is_finished = False
        # print("[index.FaissRawIndex]: shape of feature vector indexing: ", feature_vec.shape)
        self._faiss_index.add(feature_vec) #add the representation to index
        
    def search(self, query_feature_vec, k_top = 5):
        _, I = self._faiss_index.search(query_feature_vec, k_top)
        return I[0][:]
class FaissLSHIndex(FaissRawIndex):
    def __init__(self, feature_size, hash_width) -> None:
        super().__init__(feature_size)
        self._faiss_index = faiss.IndexLSH(feature_size, hash_width)
class AnnoyLSHIndex(MyIndex):
    # f_len : length of feature need to be indexed
    def __init__(self, f_len, num_tree = 10) -> None:
        super().__init__()
        self.num_tree = num_tree
        self._index = AnnoyIndex(f_len, 'hamming') # Create object AnnoyIndex use angular similarity
        self.idx = 0
    def add(self, feature_vec):
        # print("[index]: length of feature_vec = ", len(feature_vec[0]))
        self._index.add_item(self.idx, feature_vec[0])
        self.idx = self.idx + 1
    def saveIndex(self, path):
        self._index.build(self.num_tree)
        self.idx = 0
        self._index.save(path + ".ann")
    def loadIndex(self, path):
        self._index.load(path + ".ann")
    def search(self, query_feature_vec, k_top = 5):
        similar_image_indices = self._index.get_nns_by_vector(query_feature_vec[0], k_top)
        return similar_image_indices
    
    