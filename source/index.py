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

class NewLSHIndex(MyIndex):
    def __init__(self, bitdepth, input_dim, hashtable = 1) -> None:
        super().__init__()
        count = 0
        self.lsh = LSH(
                    bitdepth,
                    input_dim,
                    num_hashtables=hashtable,
                    storage_config={"dict": None}
                )
    
    def add(self, feature_vec, label):
        self.lsh.index(feature_vec, extra_data=label)
    def saveIndex(self, path):
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.lsh, f)
    def loadIndex(self, path):
        with open(path + ".pkl", "rb") as f:
            self.lsh = pickle.load(f)
    def search(self, query_feature_vec, k_top):
        points = self.lsh.query(query_feature_vec, num_results=k_top)
        (point, label), dist = points[0:k_top - 1]
        return label

class FaissRawIndex(MyIndex):
    def __init__(self, feature_size) -> None:
        super().__init__()
        self.feature_size = feature_size
        self._faiss_index = faiss.IndexFlatL2(feature_size)   # build the index
        self.is_finished = False
    def saveIndex(self, path):
        file_ver, newest_ver, isFound, path_component = self.checkIndexVer(path)
        final_path = ""
        if file_ver == 0: # no version is specified
            if (newest_ver == -1): # no index file was saved before, no conflict here
                final_path = path
            else: # save index file with greater version than every file
                final_path = path_component[0] + "-fx" + str(newest_ver + 1) + "." + path_component[1]
        else: # Save index file with specific version, so save or override
            final_path = path_component[0] + "-fx" + str(file_ver) + "." + path_component[1]
        write_index(self._faiss_index, final_path)
        self.is_finished = True
    def loadIndex(self, path):
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
        self._faiss_index = read_index(final_path)
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
    
    