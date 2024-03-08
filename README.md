# Step 1: Environment preparation:
## Create these folder and put them in project folder for running progress:
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
# Step 3: Configure the program (only for some specific tests)
Configure the evaluation mode in file config.ini

# Step 4: Run
## Command Line Arguments
### For testing:

| Flag                       | Description                                  |
| -----------                | -----------                                  |
|`--dbname <database name>`  |Specify the test database name                |
|`--dbpath <path>`           |Path to the test database                     |
|`--querypath <path>`        |Path to the query image folder                |
|`--usehash`                 |Enable to perform indexing with hash code     | 
|`--timeprior`               |Enable to use lightweight feature extractor   |
|`--multitest`               |Enable multi testing                          |
### For evaluating:
| Flag                       | Description                                  |
| -----------                | -----------                                  |
|`--evalmode`                |Enable Evaluation mode|
### For all case (must have, choose at least one):
| Flag                       | Description                                  |
| -----------                | -----------                                  |
|`--index`                   |Enable to perform indexing progress           |
|`--retrieve`                |Enable to perform query progress              |
## Run in evaluation mode
Run file main.py with flag --evalmode to evaluate the system.   
`python main.py --evalmode --index --retrieve`
## Run in test mode
Run file main.py without flag --evalmode   
`python main.py --dbname "dbname" --dbpath "path-to-db" --querypath "path-to-query-folder" --usehash --index --retrieve`

Note: To perform just retrieving, make sure the database name follow `--dbname` is not changed!


