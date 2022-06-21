# CPJKU Submission for DCASE 2022

This Repository is dedicated to the CPJKU submission of DCASE 2022, Task 1, Low-Complexity Acoustic Scene Classification.

The skeleton of the code is similar to [this repository](https://github.com/kkoutini/cpjku_dcase20).

Authors of the code:
- Florian Schmid 
- Shahed Masoudian
- Khaled Koutini 


# Preparing Environment:


An installation of [conda](https://docs.conda.io/en/latest/miniconda.html) is required on your system.

This repo uses forked versions of [sacred](https://github.com/kkoutini/sacred) for configuration and logging, [pytorch-lightning](https://github.com/kkoutini/pytorch-lightning) as a convenient pytorch wrapper and [ba3l](https://github.com/kkoutini/ba3l) as an integrating tool 
between mongodb, sacred and pytorch lightning.

-----------------------

To setup the environment [Mamba](https://github.com/mamba-org/mamba) can be used which works faster than conda:


```
conda install mamba -n base -c conda-forge
```

Now you can import the environment from environment.yml

```
mamba env create -f environment.yml
```

CAUTION: This might take several minutes.

Activate the environment:

```
conda activate dcase22_t1
```


Alternative approach is to create arbitrary environment and use _requirements.txt_ to install the packages

```
conda create -n <name> python=3.9
conda activate <name>
pip install -r requirements.txt
```
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

an example of a running code:

```
python -m experiments.dcase22.t1.teachers_gang with cp_mini_resnet models.net.rho=8 basedataset.audio_processor.sr=32000 basedataset.audio_processor.resample_only=True dcase22_reassembled soft_targets_weight=50 soft_targets_weight_long=3.0 temperature=3 random_sec=1 mixup_alpha=0.3 quantization=1 models.teacher_net.teachers_list='[228, 229]' models.net.s2_group=2 models.net.cut_channels_s3=36 teacher_long_run_id=278 
```









