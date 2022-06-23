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

Firstly, you need to create a reassembled version of the [TAU Urban Acoustic Scenes 2022 Mobile development dataset](TAU Urban Acoustic Scenes 2022 Mobile). A short draft of how to reassemble the downloaded files is provided in [files_reassemble.ipynb](files_reassemble.ipynb).

Secondly, you need to change all paths specfied in the [dataset file](datasets/dcase22/dcase22t1.py) to the correct locations on your system.

Thirdly, you need to download pre-trained PaSST models and 10-second teacher predictions from the github release provided in this repository and put them in the folder [teacher_models](teacher_models).

# Running Code:

After downloading the dataset and setting up the path to the dataset in **datasets/dcase22/dcase22t1.py** and **dcase22t1_as.py**, the terminal can be used to run the commands:

**models.net** is used to set up different configs of the model:

**basedataset.audio_processor** is used to setup different configs for the dataset processor

Custom configs can be written in **experiments/dcase22/t1/config_updates.py** in the format of dictionaries.
These configs can be called by their names.

## examples:

**models.net.rho** sets the receptive field of the network. default: 4

**models.net.s2_group** sets the grouping of convolution. default: 1 

**basedataset.audio_processor.sr**: sets the sampling rate of the raw audio file, default: 44100

**basedataset.audio_processor.resample_only**: sets the preprocessor to only resample the audio file 

**soft_targets_weight_long**: weight for the distillation loss 

**temperature:** configures the softness for the soft targets

**random_sec:** selects random 1 second from 10 seconds of audio

**mixup_alpha:** >0 augments audio file using mixup 

**mixstyle_alpha:** >0 augments audio using mixstyle 

**quantization:** =1 at the end of training performs quantization and returns final accuracy and loss based on quantized model

**models.teacher_net.teachers_list**: list of pretrained teacher for ensembling, repo contains 2 sample teachers

**cp_mini_resnet:** named config to set width, depth and rho of cp_resnet as well as weight decay of the network

# Sample Commands

## Similar to Submission 1 (t10sec) (rho=8,T=1, 10-second teacher, mixup) 
```
CUDA_VISIBLE_DEVICES=1 python -m experiments.dcase22.t1.teachers_gang with cp_mini_resnet models.net.rho=8 soft_targets_weight=50 soft_targets_weight_long=3.0 temperature=1 mixup_alpha=0.3 quantization=1 models.teacher_net.teachers_list='[253, 254, 255, 256]' models.net.s2_group=2 models.net.cut_channels_s3=36 basedataset.subsample=40 trainer.max_epochs=20
```

## Similar to Submission 2 (mixstyleR8) (rho=8, T=1, mixstyle_alpha=0.3, mixstyle_p=0.6)

```
CUDA_VISIBLE_DEVICES=1 python -m experiments.dcase22.t1.teachers_gang with cp_mini_resnet models.net.rho=8 soft_targets_weight=50 temperature=1 mixstyle_alpha=0.3 mixstyle_p=0.6 quantization=1 models.teacher_net.teachers_list='[253, 254, 255, 256]' models.net.s2_group=2 models.net.cut_channels_s3=36 trainer.max_epochs=1 
```










