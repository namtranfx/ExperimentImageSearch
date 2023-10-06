import faiss
from faiss import write_index, read_index
from annoy import AnnoyIndex
import os


class MyIndex:
    """
    The index filename has format of: metadata-fx<version>.<index file extension>
    Ex: myindex-fx5.index

    If you save a index that doesnot specify version, we will find and save as newest version
    
    If you save a index that have version existed in current folder, we will override old file,
    so BE CAREFUL WHEN INDEX DATABASE WITH A SPECIFIC VERSION OF INDEX
    
    If you save a index that have version greater much than newest version in current folder, we
    will skip the gap version

    If you load a index that doesnot have specific version, we will load the newest version in 
    current folder

    If you load a index that have specific version, we will load that right version

    """
    def __init__(self) -> None:
        self.is_created = False
        self.is_loaded = False
    def saveIndex(self, path):
        """
        Preprocess the path before saving index file 

        Need to re-implement this method for specific index
        """
        file_ver, newest_ver, isFound, path_component = self.checkIndexVer(path)
        final_path = ""
        if file_ver == 0: # no version is specified
            if (newest_ver == -1): # no index file was saved before, no conflict here
                final_path = path
            else: # save index file with greater version than every file
                final_path = path_component[0] + "-fx" + str(newest_ver + 1) + "." + path_component[1]
        else: # Save index file with specific version, so save or override
            final_path = path_component[0] + "-fx" + str(file_ver) + "." + path_component[1]
        self.final_path = final_path
    def loadIndex(self, path):
        """
        Preprocess the path before loading index file. 

        Need to re-implement this method for specific index
        """
        file_ver, newest_ver, isFound, path_component = self.checkIndexVer(path)
        print(file_ver, newest_ver, isFound)
        final_path = ""
        if file_ver == 0: # No version is specified, load newest version index file which was saved
            if (newest_ver == -1): 
                final_path = None
                print("[WARNING]: Finding the index file: ", path)
                print("[ERROR]: NO INDEX FILE FOUND!")
                exit(-100)
            elif newest_ver == 0: final_path = path
            else: final_path = path_component[0] + "-fx" + str(newest_ver) + "." + path_component[1]

        else: # Load the specific version file or load newest version file if not found
            if isFound:
                final_path = path_component[0] + "-fx" + str(file_ver) + "." + path_component[1]
            else:
                print("[WARNING]: No specific version found!") 
                print("[WARNING]: Loading newest version file...") 
                final_path = path_component[0] + "-fx" + str(newest_ver) + "." + path_component[1]
        self.final_path = final_path
    def add(self, feature_vec):
        pass
    # Return an array of image indices
    def search(self, query_feature_vec, k_top):
        pass
    def checkIndexVer(self, path) -> [int, int, bool, list]:
        """
        Check version of file in path
        
        Return value: [file_ver, newest_ver, isFound, path_component]

                file_ver: version of file (specific in path parameter)

                newest_ver: newest version of current file which was saved in folder that it's going
        to be saved in

                isFound: check if current file is existed in files in current folder 

                path_component: include folder path, filename and file extension (without version)
        """
        # Get folder path
        path_component = path.split(os.sep)
        dir_component = path_component[:-1]
        dir_path = dir_component[0]
        for component_id in range(1, len(dir_component), 1):
            dir_path = os.path.join(dir_path, dir_component[component_id])

        # Check version of current file
        file_ver = 0
        index_filename, file_extension = path_component[-1].split(".")
        # index_filename_component only have 1 or 2 component
        index_filename_component = index_filename.split("-fx") 
        if len(index_filename_component) > 1:
            file_ver = int(index_filename_component[1])
        else: file_ver = 0

        
        # Check version of file in directory
        newest_ver = -1 # No index file existed
        isFound = False # State of finding index file in directory
        print("finding index file in folder: ", dir_path)
        for filename in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, filename)):
                #print("checking file: ", filename)
                name, extension = filename.split(".")
                #name_component only have 1 or 2 component
                name_component = name.split("-fx") 
                #print("comparing two name: ", str(name_component[0]) , " and ", str(index_filename_component[0]))
                if str(name_component[0]) == str(index_filename_component[0]) and extension == file_extension:
                    print("Found file")
                    if newest_ver == -1: newest_ver = 0 # means empty version in index filename
                    if len(name_component) > 1:
                        cur_ver = int(name_component[1])
                        if cur_ver > newest_ver: newest_ver = cur_ver
                        if cur_ver == file_ver: isFound = True
        return file_ver, newest_ver, isFound, [os.path.join(dir_path, index_filename_component[0]), file_extension]

# import dependencies for this class
from sparselsh import LSH
from scipy.sparse import csr_matrix
import pickle
from lshashpy3 import LSHash
import math
class NewLSHIndex_old(MyIndex):
    def __init__(self, bitdepth, input_dim, hashtable = 5) -> None:
        super().__init__()
        self.idx = 0
        self.bitdepth = bitdepth
        self.input_dim = input_dim
        self.hash_table = hashtable

        # hashtable_t = int(math.sqrt(input_dim)) if hashtable == 0 else hashtable
        self.lsh = LSHash(hash_size=bitdepth, input_dim=input_dim, num_hashtables= hashtable)
        

    def add(self, feature_vec):
        if self.is_loaded == True:
            print("[WARNING]: Adding to the existed index. Deleting and re-initializing...")
            del self.lsh
            self.__init__(self.bitdepth, self.input_dim, self.hash_table)
            self.is_loaded = False    
        # print("index vector shape: ", feature_vec.shape)
        self.lsh.index(feature_vec.flatten(), extra_data=self.idx)
        self.idx = self.idx + 1
    
    def saveIndex(self, path):
        super().saveIndex(path)
        with open(self.final_path, "wb") as f:
            pickle.dump(self.lsh, f)
        self.is_loaded = True
    
    def loadIndex(self, path):
        if self.is_loaded == True:
            print("[WARNING]: Another index existed. Overriding new index!")
            del self.lsh
        super().loadIndex(path)
        with open(self.final_path, "rb") as f:
            self.lsh = pickle.load(f)

    def search(self, query_feature_vec, k_top):
        if self.is_loaded == False:
            print("[WARNNG]: Querying in a empty index!")
            return [None * k_top]
        nearest_vectors = self.lsh.query(query_feature_vec.flatten(), num_results=k_top, distance_func="l1norm")
        print("Number of retrieved elements: ", len(nearest_vectors))
        nearest_indices = [vec[0][1] for vec in nearest_vectors]
        return nearest_indices

class NewLSHIndex(MyIndex):
    def __init__(self, bitdepth, input_dim, hashtable = 5) -> None:
        super().__init__()
        self.idx = 0
        self.bitdepth = bitdepth
        self.input_dim = input_dim
        self.hash_table = hashtable
        self.lsh = LSH(
                    bitdepth,
                    input_dim,
                    num_hashtables=hashtable,
                    storage_config={"dict": None}
                )

    def add(self, feature_vec, label = None):
        if self.is_loaded == True:
            print("[WARNING]: Adding to the existed index. Deleting and re-initializing...")
            del self.lsh
            self.__init__(self.bitdepth, self.input_dim, self.hash_table)
            self.is_loaded = False
        # self.lsh.index(feature_vec, extra_data=label)
        self.lsh.index([feature_vec.flatten()], extra_data=self.idx)
        self.idx = self.idx + 1
    def saveIndex(self, path):
        super().saveIndex(path)
        with open(self.final_path, "wb") as f:
            pickle.dump(self.lsh, f)
        self.is_loaded = True

    def loadIndex(self, path):
        if self.is_loaded == True:
            print("[WARNING]: Another index existed. Overriding new index!")
            del self.lsh
        super().loadIndex(path)
        with open(self.final_path, "rb") as f:
            self.lsh = pickle.load(f)
        self.is_loaded = True

    def search(self, query_feature_vec, k_top):
        if self.is_loaded == False:
            print("[WARNNG]: Querying in a empty index!")
            return [None * k_top]
        points = self.lsh.query(query_feature_vec, num_results=k_top, distance_func="hamming")
        index_points = [point[0][1] for point in points]
        return index_points

class FaissRawIndex(MyIndex):
    def __init__(self, feature_size) -> None:
        super().__init__()
        self.feature_size = feature_size
        self._faiss_index = faiss.IndexFlatL2(feature_size)   # build the index
        self.is_created = True
    
        
    def saveIndex(self, path):
        super().saveIndex(path)
        write_index(self._faiss_index, self.final_path)
        self.is_loaded = True
    def loadIndex(self, path):
        if self.is_loaded == True:
            print("[WARNING]: Another index existed. Overriding new index!")
            self._faiss_index.reset()
        super().loadIndex(path)
        self._faiss_index = read_index(self.final_path)
        self.is_loaded = True
    def add(self, feature_vec):
        if self.is_loaded == True:
            print("[WARNING]: Adding to the existed index. Deleting and re-initializing...")
            self._faiss_index.reset()
            self.is_loaded = False
        # print("[index.FaissRawIndex]: shape of feature vector indexing: ", feature_vec.shape)
        self._faiss_index.add(feature_vec) #add the representation to index
        
    def search(self, query_feature_vec, k_top = 5):
        if self.is_loaded == False:
            print("[WARNNG]: Querying in a empty index!")
            return [None * k_top]
        _, I = self._faiss_index.search(query_feature_vec, k_top)
        return I[0][:]
class FaissLSHIndex(FaissRawIndex):
    def __init__(self, feature_size, hash_width) -> None:
        super().__init__(feature_size)
        self._faiss_index = faiss.IndexLSH(feature_size, hash_width)
class CustomAnnoyIndex(MyIndex):
    # f_len : length of feature need to be indexed
    def __init__(self, f_len, num_tree = 10) -> None: 
        super().__init__()
        self.feature_len = f_len
        self.num_tree = num_tree
        self._index = AnnoyIndex(f_len, 'hamming') # Create object AnnoyIndex use angular similarity
        self.idx = 0
    def add(self, feature_vec):
        # print("[index]: length of feature_vec = ", len(feature_vec[0]))
        if self.is_loaded == True:
            print("[WARNING]: Adding to the existed index. Deleting and re-initializing...")
            del self.lsh
            self.__init__(self.feature_len, self.num_tree)
            self.is_loaded = False
        self._index.add_item(self.idx, feature_vec.flatten())
        self.idx = self.idx + 1
    def saveIndex(self, path):
        super().saveIndex(path)
        self._index.build(self.num_tree)
        self.idx = 0
        self._index.save(path + ".ann")
        self.is_loaded = True
    def loadIndex(self, path):
        if self.is_loaded == True:
            print("[WARNING]: Another index existed. Overriding new index!")
            del self.lsh
            self.__init__(self.feature_len, self.num_tree)
        super().loadIndex(path)
        self._index.load(path + ".ann")
        self.is_loaded = True
    def search(self, query_feature_vec, k_top = 5):
        if self.is_loaded == False:
            print("[WARNNG]: Querying in a empty index!")
            return [None * k_top]
        similar_image_indices = self._index.get_nns_by_vector(query_feature_vec.flatten(), k_top)
        return similar_image_indices
    
    