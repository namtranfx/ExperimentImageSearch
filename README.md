# ExperimentImageSearch
An implementation of image search
# Project Structure
ExperimentImageSearch
|--data_handle
|
|
|--test
    |--cbir_flower
    |   |--requirements.txt
    |   |--run.py
    |--cbir_tiny
        |--requirements.txt
        |--run.py

# Project Requirements
Intergrated in python:
   package_name  |  use in
    functools       (2)
    time            (2)
    random          (2)(1)
    os              (2)(1)
    logging         (2)
    itertools       (1)
    zipfile         (1)
Global: 
    numpy
    tqdm
    matplotlib

(1)CBIR_FLOWER:
    pytorch
    torchvision
    glob
    pandas
    scikit-learn
    pillow
    opencv-python
    faiss
(2)CBIR_TINY:
    keras
    python-annoy
    
