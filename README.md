# CPJKU Submission for DCASE 2022

This Repository is dedicated to the CPJKU submission of DCASE 2022, Task 1, Low-Complexity Acoustic Scene Classification. It provides a lightweight version of the code we used to create our challenge submissions. The corresponding technical report will be linked as soon as it is published.

The skeleton of the code is similar to [previous CPJKU submissions](https://github.com/kkoutini/cpjku_dcase20) and the [PaSST](https://github.com/kkoutini/PaSST) repository.

Authors of the code:
- [Florian Schmid](https://github.com/fschmid56/)
- [Shahed Masoudian](https://github.com/ShawMask)
- [Khaled Koutini](https://github.com/kkoutini) 


# Setting up the Environment:


An installation of [conda](https://docs.conda.io/en/latest/miniconda.html) is required on your system.

This repo uses forked versions of [sacred](https://github.com/kkoutini/sacred) for configuration and logging, [pytorch-lightning](https://github.com/kkoutini/pytorch-lightning) as a convenient pytorch wrapper and [ba3l](https://github.com/kkoutini/ba3l) as an integrating tool 
between mongodb, sacred and pytorch lightning.

-----------------------

To setup the environment [Mamba](https://github.com/mamba-org/mamba) is recommended and faster than conda:


```
conda install mamba -n base -c conda-forge
```

Now you can import the environment from environment.yml. This might take several minutes.

```
mamba env create -f environment.yml
```

Alternatively, you can also import the environment using conda:

```
conda env create -f environment.yml
```

An environment named `dcase22_t1` has been created. Activate the environment:

```
conda activate dcase22_t1
```


Now install `sacred`, `ba3l` and `pl-lightening`:

```shell
# dependencies
pip install -e 'git+https://github.com/kkoutini/ba3l@v0.0.2#egg=ba3l'
pip install -e 'git+https://github.com/kkoutini/pytorch-lightning@v0.0.1#egg=pytorch-lightning'
pip install -e 'git+https://github.com/kkoutini/sacred@v0.0.1#egg=sacred' 
```

# Setting up the external data resources:

Firstly, you need to create a reassembled version of the [TAU Urban Acoustic Scenes 2022 Mobile development dataset](https://zenodo.org/record/6337421#.YrQaohuxVhE). A short draft of how to reassemble the downloaded files is provided in [files_reassemble.ipynb](files_reassemble.ipynb).

Secondly, you need to change all paths specfied in the [dataset file](datasets/dcase22/dcase22t1.py) to the correct locations on your system.

Thirdly, you need to download pre-trained PaSST models and 10-second teacher predictions from the [github release](https://github.com/CPJKU/cpjku_dcase22/releases/tag/v0.0.1) provided in this repository and put them in the folder [teacher_models](teacher_models).

# Running Experiments:

After creating the environment and setting up external data resources the following is the simplest command to run:

```
python -m experiments.dcase22.t1.teachers_gang
```
And a short test run:

```
python -m experiments.dcase22.t1.teachers_gang with trainer.max_epochs=10 basedataset.subsample=100
```

-----------------------

A lot configuration possibilities can be set via the command line, here are examples of the configurations we used in our submissions:

**Network configuration**:

- **cp_mini_resnet:** named config to decrease the number of blocks in the network
- **models.net.rho:** changes receptive field of the network
- **models.net.s2_group:** grouping in second network stage
- **models.net.cut_channels_s3:** cutting the width on stage 3 of the network

**Knowledge Distillation**:

- **temperature:** temperature used to create soft targets
- **soft_targets_weight:** weighting of the distillation loss
- **models.teacher_net.teachers_list:** specifies list of teacher ids to be ensembled
- **soft_targets_weight_long:** weighting of the distillation loss reagrding a 10-second teacher

**Mixup and Mixstyle**:

- **mixup_alpha**: mixing coefficient for mixup
- **mistyle_alpha**: mixing coefficient for mixstyle
- **mixstyle_p**: probability of mixstyle being applied to a batch

**Quantization**:

- **quantization**: setting quantization=1 evaluates the quantized model on the test split of the development set after training


# Submission Commands

## Similar to Submission 1 (t10sec) (rho=8, T=1, 10-second teacher, mixup) 
```
python -m experiments.dcase22.t1.teachers_gang with cp_mini_resnet models.net.rho=8 soft_targets_weight=50 soft_targets_weight_long=3.0 temperature=1 mixup_alpha=0.3 quantization=1 models.teacher_net.teachers_list='[253, 254, 255, 256]' models.net.s2_group=2 models.net.cut_channels_s3=36 
```

## Similar to Submission 2 (mixstyleR8) (rho=8, T=1, mixstyle_alpha=0.3, mixstyle_p=0.6)

```
python -m experiments.dcase22.t1.teachers_gang with cp_mini_resnet models.net.rho=8 soft_targets_weight=50 temperature=1 mixstyle_alpha=0.3 mixstyle_p=0.6 quantization=1 models.teacher_net.teachers_list='[253, 254, 255, 256]' models.net.s2_group=2 models.net.cut_channels_s3=36 
```

## Using MongoDB for Logging

Install pymongo:

```
pip install pymongo
```

If you have a running MongoDB-Server you can append the following to your command to log experiment details:

```
-p -m mongodb_server:[port]:[name] -c "Testing CPJKU Submission to DCASE22"
```








