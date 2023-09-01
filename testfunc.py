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

import torch
from torchvision.models import resnet50, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


# To assist you in designing the feature extractor you may want to print out
# the available nodes for resnet50.
m = mobilenet_v3_small()
train_nodes, eval_nodes = get_graph_node_names(mobilenet_v3_large())
print(train_nodes)
exit()
# The lists returned, are the names of all the graph nodes (in order of
# execution) for the input model traced in train mode and in eval mode
# respectively. You'll find that `train_nodes` and `eval_nodes` are the same
# for this example. But if the model contains control flow that's dependent
# on the training mode, they may be different.

# return_nodes = {
#     'flatten':'layer1'
# }

# To specify the nodes you want to extract, you could select the final node
# that appears in each of the main layers:
# return_nodes = {
#     # node_name: user-specified key for output dict
#     'layer1.2.relu_2': 'layer1',
#     'layer2.3.relu_2': 'layer2',
#     'layer3.5.relu_2': 'layer3',
#     'layer4.2.relu_2': 'layer4',
# }

# But `create_feature_extractor` can also accept truncated node specifications
# like "layer1", as it will just pick the last node that's a descendent of
# of the specification. (Tip: be careful with this, especially when a layer
# has multiple outputs. It's not always guaranteed that the last operation
# performed is the one that corresponds to the output you desire. You should
# consult the source code for the input model to confirm.)
# return_nodes = {
#     'layer1': 'layer1',
#     # 'layer2': 'layer2',
#     # 'layer3': 'layer3',
#     # 'layer4': 'layer4',
# }

# Now you can build the feature extractor. This returns a module whose forward
# method returns a dictionary like:
# {
#     'layer1': output of layer 1,
#     'layer2': output of layer 2,
#     'layer3': output of layer 3,
#     'layer4': output of layer 4,
# }
#create_feature_extractor(m, return_nodes=return_nodes)
return_nodes = {
    'flatten':'final_feature'
}
body = create_feature_extractor(mobilenet_v3_small(), return_nodes=return_nodes)

inp = torch.randn(1, 3, 224, 224)
out = body(inp)
print(out['final_feature'].shape)
print(type(out))
exit()
# Let's put all that together to wrap resnet50 with MaskRCNN

# MaskRCNN requires a backbone with an attached FPN
class Resnet50WithFPN(torch.nn.Module):
    def __init__(self):
        super(Resnet50WithFPN, self).__init__()
        # Get a resnet50 backbone
        m = resnet50()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


# Now we can build our model!
model = MaskRCNN(Resnet50WithFPN(), num_classes=91).eval()