# from torchvision.datasets import Caltech101
# from torch.utils.data import random_split

# data = Caltech101(root='./data', download=True)
# train_size = int(0.8 * len(data))
# test_size = len(data) - train_size
# train_data, test_data = random_split(data, [train_size, test_size])

# image, target = train_data[1500]
# print("Label of the 1500 element: ", target)

# #==================================================================================
# from datasketch import WeightedMinHashGenerator

# # Create WeightedMinHashGenerator
# wmg = WeightedMinHashGenerator(features.shape[1], sample_size=128)

# # Create MinHash objects for each feature vector
# minhashes = [wmg.minhash(features[i]) for i in range(features.shape[0])]

# # Create LSH index with 128-bit hash values
# lsh = MinHashLSH(num_perm=128) # num_perm = number of bit
# for i, minhash in enumerate(minhashes):
#     lsh.insert(i, minhash)

# # Query LSH index for nearest neighbors
# result = lsh.query(minhashes[0])

# #=======================version using SimHash
# from datasketch import WeightedMinHashGenerator

# # Create WeightedMinHashGenerator
# wmg = WeightedMinHashGenerator(features.shape[1], sample_size=128)

# # Create MinHash objects for each feature vector
# minhashes = [wmg.minhash(features[i]) for i in range(features.shape[0])]

# # Create LSH index with 128-bit hash values
# lsh = MinHashLSH(num_perm=128)
# for i, minhash in enumerate(minhashes):
#     lsh.insert(i, minhash)

# # Query LSH index for nearest neighbors
# result = lsh.query(minhashes[0])
# #==========================version which save and load index file
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input
# from keras.preprocessing import image
# from datasketch import WeightedMinHashGenerator
# from scipy.spatial.distance import cdist
# import numpy as np

# # Load pre-trained ResNet50 model
# model = ResNet50(weights='imagenet', include_top=False)

# # Load and preprocess image
# img_path = 'new_image.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# # Extract deep features using ResNet50
# features = model.predict(x)

# # Flatten features
# features = features.reshape((features.shape[0], -1))

# # Create WeightedMinHashGenerator
# wmg = WeightedMinHashGenerator(features.shape[1], sample_size=128)

# # Compute MinHash for query feature vector
# query_minhash = wmg.minhash(features[0])

# # Convert MinHash to numpy array
# query_hashcode = np.array(query_minhash.hashvalues)

# # Load hashcodes from file
# hashcodes = np.load('hashcodes.npy')

# # Compute Hamming distances between query hashcode and all other hashcodes
# hamming_distances = cdist(query_hashcode.reshape(1, -1), hashcodes, metric='hamming')

# # Find indices of nearest neighbors
# k = 3
# nn_indices = np.argpartition(hamming_distances, k)[:, :k]
# #=====================this is version that use lsh.query instead of calc hamming distance manually
# from datasketch import MinHashLSH

# # Load hashcodes from file
# hashcodes = np.load('hashcodes.npy')

# # Create MinHash objects from hashcodes
# minhashes = [MinHash(seed=0, hashvalues=hashcode) for hashcode in hashcodes]

# # Create LSH index with 128-bit hash values
# lsh = MinHashLSH(num_perm=128)
# for i, minhash in enumerate(minhashes):
#     lsh.insert(i, minhash)

# # Query LSH index for nearest neighbors
# result = lsh.query(minhashes[0])
# #=========================
# """
# difference between MinHash and SimHash in implementation:
#     - MinHash: use MinHash and MinHashLSH
#     - SimHash: use WeightedMinHashGenerator and MinHashLSH
# MinHash is used to approx. Jaccard similarity
# SimHash is used to approx. Cosine similarity

# sample_size is parameter of MinHash func, isnot size of deep feature.

# """
# #=====================================
# from annoy import AnnoyIndex
# f = 40 # Chiều dài của vector sẽ được lập chỉ mục
# t = AnnoyIndex(f, 'angular') # Khởi tạo đối tượng AnnoyIndex
# # Thêm các vector đặc trưng vào chỉ mục
# for i in range(1000):
#     v = [random.gauss(0, 1) for z in range(f)]
#     t.add_item(i, v)

# t.build(10) # Xây dựng cây với 10 cây
# t.save('test.ann') # Lưu chỉ mục xuống đĩa

# # Tải lại chỉ mục từ đĩa
# u = AnnoyIndex(f, 'angular')
# u.load('test.ann')

# # Tìm 1000 hàng xóm gần nhất của phần tử 0
# print(u.get_nns_by_item(0, 1000))

# #============================= KB-tree and Hierachiecal-tree Indexing
# from scipy import spatial
# import numpy as np
# from sklearn.neighbors import NearestNeighbors

# # Tạo dữ liệu mẫu
# data = np.random.rand(1000, 10)

# # Đánh chỉ mục dữ liệu sử dụng cây KD trong scipy
# tree = spatial.KDTree(data)

# # Truy vấn các điểm lân cận gần nhất
# distances, indices = tree.query([data[0]], k=5)

# # Đánh chỉ mục dữ liệu sử dụng cây phân cấp trong scikit-learn
# nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(data)

# # Truy vấn các điểm lân cận gần nhất
# distances, indices = nbrs.kneighbors([data[0]])

# #==========save  and load 
# from sklearn.externals import joblib

# # Lưu cây KD xuống bộ nhớ
# joblib.dump(tree, 'kd_tree.pkl')

# # Tải cây KD từ bộ nhớ
# tree = joblib.load('kd_tree.pkl')

# # Lưu cây phân cấp xuống bộ nhớ
# joblib.dump(nbrs, 'ball_tree.pkl')

# # Tải cây phân cấp từ bộ nhớ
# nbrs = joblib.load('ball_tree.pkl')


# import torch
# from source.TinyViT.tiny_vit import TinyViT, tiny_vit_21m_224, tiny_vit_21m_384
# # Định nghĩa một hàm forward hook để lấy giá trị đầu ra của stage cuối cùng
# def get_last_stage_output(module, input, output):
#     global last_stage_output
#     last_stage_output = output

# class MyModel:
#     def __init__(self):
#         # Tạo mô hình TinyViT-21M
#         # self.model = TinyViT()
        
#         self.model = tiny_vit_21m_224(pretrained=True)
#         print("[check]: model is instance of TinyViT? ", isinstance(self.model, TinyViT))
#         # Tải trọng số đã được huấn luyện sẵn
#         #self.model.load_state_dict(torch.load('tiny_vit_21m_22kto1k_distill.pth'))
#     def extract(self, x):    
#         return self.model.forward_features(x)
        
        
# from source.features import MyEfficientViT

# test = MyEfficientViT()

