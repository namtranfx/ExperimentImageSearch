# Step 1: Environment preparation:
## Create these folder for running progress:
    ./index
    ./log
    ./result
## Install requirement packages/libraries in file: manual_requirements.txt
    pip install -r manual_requirements.txt

## Manually install torch and torchvision if not available (not necessary when run on kaggle)
    pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


# Step 2: Prepare dataset:
## Get dataset from following link:
    https://www.kaggle.com/datasets/natra2k/excbir
## Update dataset folder path in configuration file (as in step 3)
Note: Fixed within 4 dataset: caltech101, cifar10, oxford102flower, coco2017
# Step 3: Configure the program
Configure the evaluation mode in file config.ini

# Step 4: Run

Run file main.py to do both index image database and evaluate the query process



