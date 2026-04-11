# Introduction

This repository is forked from: https://github.com/hcai-mms/jam

## Citation
It reproduces the research paper:
```bibtex
@inproceedings{melchiorre2025jam,
  title     = {Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation},
  author    = {Alessandro B. Melchiorre and Elena V. Epure and Shahed Masoudian and Gustavo Escobedo and Anna Hausberger and Manuel Moussallam and Markus Schedl},
  booktitle = {Proceedings of the 19th ACM Conference on Recommender Systems (RecSys)},
  year      = {2025},
  address   = {Prague, Czhech Republic},
  note      = {Short Paper},
  publisher = {ACM},
}
```

## Overview
This project focuses on implementing and understanding the JAM architecture, which learns joint representations of queries, users, and items for improved recommendation performance.

## My Contributions
- Reproduced the JAM model locally from the research paper 
- Understood the query-item matching mechanism 
- Used Virtual Environment instaed of Conda
- Set up training and evaluation pipeline 
- Added scripts to fetch, preprocess, split training data with `python run_preprocess.py` 
- Used the trained model to generate song recommendation for user queries with python -m usage. `python run_implementation.py -p <Path_of_folder_with_saved_model>

## Note
This work is for educational and research purposes. All credits for the original idea and architecture go to the paper authors.


# JAM Reproduction

# :jar: :cherries: JAM - Just Ask for Music  
### Multimodal and Personalized Natural Language Music Recommendation


<div align="center">
    <img src="./assets/jam_cute.png" style="width: 320px" />
</div>

## Installation & Setup

### Environment

- Python used `3.10.20`
- Install the environment with `python -m venv .venv`
- Activate the environment on windows with `.venv/Scripts/Activate.ps1`
- Install all dependies with `pip install -r requirements.txt`


### Data

- The dataset is saved on zenodo (https://zenodo.org/records/15968495)
- In order to download the dataset using python run `python data/fetch_data`
- run the associated files for the pre-processing with `python data\preprocess_data` and splitting `python data\split_data`
- a folder `processed` should have the following files (/ indicates or):
    - <train/val/test>_split.tsv
    - <user/item>_idxs.tsv
    - <user/item>_<any_modality>_features.npy

### Logging

- JAM uses [W&B](https://wandb.ai/site) for logging. You should create an account there first
- Modify the `constants/wandb_constants.py` file with your `entity_name` and `project_name`
- First time usage you might want call `wandb login` from the shell.


## Usage
General flow is
1. Create a configuration file
2. Call `run_experiments.py`

The framework will take care of:
- Loading the data
- Training/Validating (optionally Testing) the model
- Saving the best model and configuration
- Log results to W&B

### Running a Single Experiment
A single experiments can be 1) `train/val` + `test` 2) `train/val` 3) just `test`

1. Create a `.yml` config file (possibly in `conf/confs/`). See `conf/confs/template_conf.yml` for explanations of the possible values. See `constants/conf_constants.py` for defaults.
   1. Minimally, you should specify `data_path`, where `data/<dataset_name>` is looked for.
   2. Additionally, you should also add hyperparameters of your chosen algorithm.
   3. Running `test` as experiment type requires `model_path` to the saved model.
2. `python run_experiments.py -a <alg> -d <dataset> -c <path_to_conf> -t <run_type>`
   1. For `alg` and `dataset` see the available ones in `constants/enums.py`
   2. `path_to_conf` is what you specified above
   3. For `run_type` and other variables see `run_experiments.py`
3. Look at how your experiment is doing on W&B.

Example:
`test_conf.yml`
```yml
data_path: "./data"
d: 28 # for avgmatching model
device: cuda

n_epochs: 50
eval_batch_size: 256
train_batch_size: 256

running_settings:
  train_n_workers: 4
  eval_n_workers: 4
  batch_verbose: True
```
then run
`python run_experiment.py -a basematching -d zenodo -c conf/confs/test_conf.yml`
(if `-t` is not specified, it will run `train/val/test`)


### Running Multiple Experiments (Sweeps with W&B)
To run multiple experiments, JAM relies on W&B sweeps. This is to execute different `train/val` experiments.

1. Create a `.yml` config file (possibly in `conf/sweeps/`). Take `conf/sweeps/template_sweep_conf.yml` as reference. See `constants/conf_constants.py` for defaults.
      1. Specify again your `entity_name` and `project_name` in the conf. These are the same values you had for the Logging step above.
      2. Give a meaningful name to your sweep (e.g. `<algorithm_name>-<dataset_name>` should suffice. Add these values to the `parameters` section as well. See `constants/enums.py` for possible values. 
      3. Adjust the rest of the configuration as you please. See the [official docs](https://docs.wandb.ai/guides/sweeps/) on W&B.  
2. Activate the sweep. `wandb sweep conf/sweeps/test_sweep_conf.yml`. Your sweep now should be online and can be monitored on your dashboard.
3. Start 1+ agents. The command to start an agent is returned by wandb when activating the sweep. It's in the shape of `wandb agent <entity_name>/<project_name>/<sweep_id>`.

NB. You can adjust how many gpus are visible to the agent by specifying `CUDA_VISIBLE_DEVICES=... wandb agent..`

#### Multiple Runs on a Server
After activating a sweep, you can use `run_agents.py` to launch multiple agents simultaneously.

When running `run_agent.py` you need to specify:
- `sweep_id`. This value should be in the format `<entity_name>/<project_name>/<sweep_id>`
- `available_gpus` (e.g. the indexes)
- `n_parallel` or how many agents PER gpu. Need to be careful with also the # of workers.
## Extend
### Codebase Structure
```
.
├── algorithms                  <- Classes about Query-User-Item Matching
├── conf                        <- Parsing & Storing .yml conf file
├── constants                   <- Constants & Enums used across the codebase
├── data                        <- Data classes, Raw and Processed Datasets
├── evaluation                  <- Metrics and Evaluation Procedure
├── train                       <- Trainer class
├── utilities                   <- Utilities from mild to low importance
├── (saved_models)              <- Automatically created (if def. conf is not altered)
├── experiment_helper.py        <- Executes the main functionalities of the code.
├── sweep_agent.py              <- Same as experiment_helper but for train_val and when launching sweeps.
├── run_test_sweep.py           <- Same as experiment_helper but for test results over a sweep.
└── run_experiment.py           <- Entry point to the code.
```
### Add Algorithms
Take a look at the `BaseQueryMatchingModel`in `algorithms/base` on what functionalities are expected from a new algorithm.

You can implement your class in `algorithms/alg` (e.g. look at `AverageQueryMatching`). Creating a descendent of `BaseQueryMatchingModel` would be the best ;). 

When the main methods are implemented, add your class to `AlgorithmsEnum` in `constants/enums.py` so it can be recognized when calling `run_experiments`

### Add Datasets
Choose a name, short and lowercase letters to denote the dataset `<dataset_name>`

The expected format of the files are in the first lines in `data/datasets.py` (for the user-query-item matching) and `data/feature.py` (for pre-trained user/item features).

If you can provide the files in the above format, you can add them to ``data/<dataset_name>/processed``

Add your dataset to `DatasetsEnum` in `constants/enums.py`.

NB. Codebase will look for the data in `os.path.join(conf['data_path'],<dataset_name>,'processed')`

# JAMSessions Dataset

-> [Link to the Dataset on Zenodo](https://zenodo.org/records/15968495) <-

Data | Unique # of samples | Field in Dataset | Information 
 --- |---------------------|------------------|-------------|
Query ID | 112,337             | query_idx        |                                        |
Query | 112,337             | text             |
User ID | 103,752             | user_idx         | 
Playlist | 3,978               | item_idxs        | ISRC code of each item in the playlist 

### Statistics

The dataset provides information of 3,978 playlists consisting of 99,865 unique items matched with 112,337 unique queries of 103,752 users.


## License

The code in this repository is licensed under the MIT License. For details, please see the [LICENSE](./LICENSE) file.






