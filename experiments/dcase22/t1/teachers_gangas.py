from pathlib import Path
from re import L

from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from git import Repo
import random
from einops import repeat, rearrange

from experiments.dcase22.t1.config_updates import add_configs
from helpers.workersinit import worker_init_fn
from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule
from sacred.config_helpers import DynamicIngredient, CMD
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.utils import mixup, mixstyle, spawn_get
from helpers import nessi
from helpers.utils import MyStaticPostQuantizationCallback


"""
This Training code is exactly like teachers_gang but covers audioset training as well
"""


ex = Experiment("gang_as")

# define datasets and loaders
get_train_loader = ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=80,
                          num_workers=16, shuffle=True, dataset=CMD("/basedataset.get_training_set_raw"),
                          )




get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=20, num_workers=16,
                                            dataset=CMD("/basedataset.get_test_set_raw"))

ex.datasets.quantized_test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), test=True,
                                            batch_size=10, num_workers=10, dataset=CMD("/basedataset.get_test_set_raw"))

get_eval_loader = ex.datasets.evaluate.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                           batch_size=10, num_workers=10, dataset=CMD("/basedataset.get_eval_set_raw"))



# Audioset loader

audioset_train_loader = ex.datasets.audioset.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), batch_size=8,
                          num_workers=16, shuffle=None, dataset=CMD("/audioset.get_full_training_set"),
                          sampler=CMD("/audioset.get_ft_weighted_sampler"))


@ex.config
def default_conf():
    cmd = " ".join(sys.argv)
    saque_cmd = os.environ.get("SAQUE_CMD", "").strip()
    saque_id = os.environ.get("SAQUE_ID", "").strip()
    commit_hash = Repo(search_parent_directories=True).head.object.hexsha
    process_id = os.getpid()
    models = {
        # the net to be trained (student)
        "net": DynamicIngredient("models.cp.cp_resnet.model_ing"),
        # dynamically creating spectrograms for student
        # dataset implementation (right now) fetches cached signals with a single sampling rate (32k - which is required
        # by the transformer)
        # increase hop size for student from 512 to 744 (to compensate for increased sampling rate)
        "mel": DynamicIngredient("models.passt.preprocess.model_ing",
                                         instance_cmd="AugmentMelSTFT",
                                         n_mels=256, sr=32000, win_length=2048, hopsize=744, n_fft=2048, freqm=0,
                                         timem=0,
                                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                                         fmax_aug_range=1),
        # the teacher network (weights to be loaded)
        "teacher_net": DynamicIngredient("models.passt.passt.model_ing", instance_cmd="get_teacher_avg_ensemble", teachers_list=[]),
        # dynamically creating spectrograms for teacher
        "teacher_mel": DynamicIngredient("models.passt.preprocess.model_ing",
                                 instance_cmd="AugmentMelSTFT",
                                 n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                                 timem=20,
                                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                                 fmax_aug_range=1000)
    }
    basedataset = DynamicIngredient(
        "datasets.dcase22.dcase22t1.dataset", audio_processor=dict(sr=32000,resample_only=True))

    audioset = DynamicIngredient("datasets.audioset.dataset", wavmix=1)

    trainer = dict(max_epochs=750, gpus=1,
                   weights_summary='full', benchmark=True)
    device_ids = {'a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6'}
    lr = 0.0001

AS_loader=None


# command to get one batch from audio set
@ex.command
def get_audioset_training_batch():
    global AS_loader
    try:
        X, f, Y = next(AS_loader)
    except:
        AS_loader = iter(audioset_train_loader())
        X, f, Y = next(AS_loader)
    return X, f, Y

# register extra possible configs
add_configs(ex)



@ex.command
def get_scheduler_lambda(warm_up_len=100, ramp_down_start=250, ramp_down_len=400, last_lr_value=0.005,
                         schedule_mode="exp_lin"):
    """
    @param warm_up_len: number of epochs for the lr to reach its maximum value
    @param ramp_down_start: control the epoch where decline of the lr starts
    @param ramp_down_len: number of epochs it takes for the lr to descend
    @param last_lr_value: final value of lr as a percentage to the original lr
    @param schedule_mode: method of scheduling 'exp_lin' and 'cos_cyc' are available
    @return: new value for the lr
    """
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    if schedule_mode == "cos_cyc":
        return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda funtion.")


@ex.command
def get_lr_scheduler(optimizer, schedule_mode):
    """
      @param optimizer: optimizer used for training
      @param schedule_mode: scheduling mode of the lr
      @return: updated version of the optimizer with new lr
      """
    if schedule_mode in {"exp_lin", "cos_cyc"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler_lambda())
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


@ex.command
def get_optimizer(params, lr, weight_decay=0.001):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


class M(Ba3lModule):
    def __init__(self, experiment):
        super(M, self).__init__(experiment)
        # all available devices
        self.device_ids = self.config.device_ids

        # classification of the devices
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # type of augmentation for mixup and mix style
        self.mixup_alpha = self.config.get("mixup_alpha", False)

        self.mixstyle_p = self.config.get("mixstyle_p", 0.0)
        self.mixstyle_alpha = self.config.get("mixstyle_alpha", 0.1)

        # randomly crop out 1/10 of the 10 second snippets from dcase21
        #  - dcase21 has 1/10 of the files of dcase22
        #  - multiply epochs by 10
        #  - extend learning rate schedule by factor of 10
        #  - results in exactly the same number of training steps as training on 1 second snippets of dcase22
        #  - for validation: split 10 second files into 1 second files and repeat labels 10 times
        self.random_sec = self.config.get("random_sec", False)

        # tempreture to control the softness of soft logits for the teacher
        self.temperature = self.config.get("temperature", 5)
        self.soft_targets_weight = self.config.get("soft_targets_weight", 100)  # weight loss for soft targets
        self.label_loss_weight = self.config.get("label_loss_weight",
                                                 0.5)  # weight for true label cross entropy loss
        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none")  # KL Divergence loss for soft targets
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # weight loss for soft targets of teacher model predictions for full 10 second snippets
        self.soft_targets_weight_long = self.config.get("soft_targets_weight_long", 0)
        if self.soft_targets_weight_long:
            pred_file = get_pred_file(run_id=self.config.get("teacher_long_run_id", 174), db_name="ast_dcase22t1")
            teacher_logits = torch.load(pred_file)
            self.teacher_soft_targets_long = self.log_softmax(teacher_logits.float() / self.temperature)

        self.quantized_mode=False
        self.do_swa = False

    # logits of the student
    def forward(self, x):
        return self.net(x)

    # logits of the teacher
    def teacher_forward(self, x):
        self.teacher_net.eval()
        return self.teacher_net(x)

    # get the output of the convolutions for the student
    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def teacher_mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.teacher_mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    # training step for the pytorch lightning
    def training_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
        as_x, _, _ = get_audioset_training_batch()
        batch_size = len(y)
        full_bath_size = batch_size+batch_size

        if self.random_sec:
            # x is of shape: batch_size, 1, 320000
            # randomly pick 1 second from 10 seconds
            samples_to_pick = x.size(2) // 10
            t_start = np.random.randint(0, x.size(2)-samples_to_pick)
            # crop one second audio
            x = x[:, :, t_start:t_start+samples_to_pick]
            as_x = as_x.to(x.device)
            #print("as_x shape ",as_x.shape)
            as_x = rearrange(
                as_x, 'b 1 (slices t) -> (b slices) 1 t', slices=10)
        x = torch.cat((x, as_x[:batch_size]), dim=0)
        x_teacher = self.teacher_mel_forward(x)
        x = self.mel_forward(x)

        

        if self.mixstyle_p > 0:
            raise NotImplementedError()
            # mixstyle not applied to teacher - think about if this is necessary???
            x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

        if self.mixup_alpha:
            rn_indices, lam = mixup(batch_size, self.mixup_alpha)
            lam = torch.cat((lam, torch.ones_like(lam)), dim=0)
            rn_indices = torch.cat(
                (rn_indices, torch.arange(batch_size, full_bath_size)), dim=0)  # don't mix audioset samples
            lam = lam.to(x.device)
            x = x * lam.reshape(full_bath_size, 1, 1, 1) + \
                x[rn_indices] * (1. - lam.reshape(full_bath_size, 1, 1, 1))
            # applying mixup also to teacher spectrograms
            x_teacher = x_teacher * lam.reshape(full_bath_size, 1, 1, 1) + \
                x_teacher[rn_indices] * \
                (1. - lam.reshape(full_bath_size, 1, 1, 1))

            y_hat = self.forward(x)

            samples_loss = (F.cross_entropy(y_hat[:batch_size], y, reduction="none") * lam[:batch_size] +
                            F.cross_entropy(y_hat[:batch_size], y[rn_indices[:batch_size]], reduction="none") * (1. - lam[:batch_size]))
        else:
            raise NotImplementedError()
            y_hat = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        label_loss = samples_loss.mean()
        samples_loss = samples_loss.detach()
        devices = [d.rsplit("-", 1)[1][:-4] for d in files]
        _, preds = torch.max(y_hat[:batch_size], dim=1)
        n_correct_pred = (preds == y).sum()

        # Getting teacher predictions
        with torch.no_grad():
            # teacher already trained - no weight updates
            y_hat_teacher, embed = self.teacher_forward(x_teacher)
            

        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_soft_teacher = self.log_softmax(y_hat_teacher / self.temperature)
            y_soft_student = self.log_softmax(y_hat / self.temperature)

        soft_targets_loss = self.kl_div_loss(y_soft_student, y_soft_teacher).mean()  # mean since reduction="none"

        if self.soft_targets_weight_long:
            raise NotImplementedError()
            # getting teacher predictions on full 10 second snippet
            y_soft_teacher_long = self.teacher_soft_targets_long[indices].to(y_hat.device)
            if self.mixup_alpha:
                soft_targets_loss_long = self.kl_div_loss(y_soft_student, y_soft_teacher_long).mean(dim=1) \
                                         * lam.reshape(batch_size) + \
                                    self.kl_div_loss(y_soft_student, y_soft_teacher_long[rn_indices]).mean(dim=1) \
                                         * (1. - lam.reshape(batch_size))
                soft_targets_loss_long = soft_targets_loss_long.mean()
            else:
                soft_targets_loss_long = self.kl_div_loss(y_soft_student, y_soft_teacher_long).mean()  # mean since reduction="none"
            loss = self.soft_targets_weight * soft_targets_loss + \
                   self.label_loss_weight * label_loss + \
                   self.soft_targets_weight_long * soft_targets_loss_long
        else:
            loss = self.soft_targets_weight * soft_targets_loss + self.label_loss_weight * label_loss

        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}

        results["teacher_loss_weighted"] = soft_targets_loss.detach() * self.soft_targets_weight
        results["label_loss_weighted"] = label_loss.detach() * self.label_loss_weight

        if self.soft_targets_weight_long:
            results["teacher_loss_long_weighted"] = soft_targets_loss_long.detach() * self.soft_targets_weight_long
        else:
            results["teacher_loss_long_weighted"] = torch.as_tensor(0., device=self.device)

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devcnt." + d] = results["devcnt." + d] + 1.

        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)

        logs = {'train.loss': avg_loss, 'train_acc': train_acc, 'step': self.current_epoch}
        avg_label_loss_weighted = torch.stack([x['label_loss_weighted'] for x in outputs]).mean()
        avg_teacher_loss_weighted = torch.stack([x['teacher_loss_weighted'] for x in outputs]).mean()
        avg_teacher_loss_long_weighted = torch.stack([x['teacher_loss_long_weighted'] for x in outputs]).mean()
        logs['label_loss_weighted'] = avg_label_loss_weighted
        logs['teacher_loss_weighted'] = avg_teacher_loss_weighted
        logs['teacher_loss_long_weighted'] = avg_teacher_loss_long_weighted

        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            logs["tloss." + d] = dev_loss / dev_cnt
            logs["tcnt." + d] = dev_cnt
        self.log_dict(logs)

    def validation_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
        if self.random_sec:
            # x is of shape: batch_size, 1, 320000
            y = repeat(y, 'b -> (b 10)')
            x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
            files = repeat(np.array(files), 'b -> (b 10)')
            device_indices = repeat(device_indices, 'b -> (b 10)')
            cities = repeat(cities, 'b -> (b 10)')
            indices = repeat(indices, 'b -> (b 10)')

        x = self.mel_forward(x)

        if self.quantized_mode:
            x, y = x.cpu(), y.cpu()
        # SWA edits
        model_name = [("", self.net)]
        if self.do_swa and not self.quantized_mode:
            model_name = model_name + [("swa_", self.net_swa)]
        all_results = {}
        for net_name, net in model_name:
            y_hat = net(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            loss = samples_loss.mean()

            self.log("validation.loss", loss, prog_bar=True,
                     on_epoch=True, on_step=False)
            _, preds = torch.max(y_hat, dim=1)
            n_correct_pred_per_sample = (preds == y)
            n_correct_pred = n_correct_pred_per_sample.sum()
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]
            results = {"val_loss": loss,
                       "n_correct_pred": n_correct_pred, "n_pred": len(y)}

            for d in self.device_ids:
                results["devloss." +
                        d] = torch.as_tensor(0., device=self.device)
                results["devcnt." +
                        d] = torch.as_tensor(0., device=self.device)
                results["devn_correct." +
                        d] = torch.as_tensor(0., device=self.device)
            for i, d in enumerate(devices):
                results["devloss." + d] = results["devloss." + d] + \
                    samples_loss[i]
                results["devn_correct." + d] = results["devn_correct." +
                                                       d] + n_correct_pred_per_sample[i]
                results["devcnt." + d] = results["devcnt." + d] + 1

            # aggregate results
            all_results = {**all_results,
                           **{net_name+k: v for k, v in results.items()}}
        return all_results

    # validation epoch for pytorch lightning
    def validation_epoch_end(self, outputs):
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = torch.stack([x[net_name+'val_loss']
                                   for x in outputs]).mean()

            val_acc = sum([x[net_name+'n_correct_pred'] for x in outputs]
                          ) * 1.0 / sum(x[net_name+'n_pred'] for x in outputs)
            logs = {'val.loss': avg_loss, 'val_acc': val_acc,
                    'step': self.current_epoch}

            for d in self.device_ids:
                dev_loss = torch.stack(
                    [x[net_name+"devloss." + d] for x in outputs]).sum()
                dev_cnt = torch.stack([x[net_name+"devcnt." + d]
                                      for x in outputs]).sum()
                dev_corrct = torch.stack(
                    [x[net_name+"devn_correct." + d] for x in outputs]).sum()
                logs["vloss." + d] = dev_loss / dev_cnt
                logs["vacc." + d] = dev_corrct / dev_cnt
                logs["vcnt." + d] = dev_cnt
                # device groups
                logs["acc." + self.device_groups[d]
                     ] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
                logs["count." + self.device_groups[d]
                     ] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
                logs["lloss." + self.device_groups[d]
                     ] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

            for d in set(self.device_groups.values()):
                logs["acc." + d] = logs["acc." + d] / logs["count." + d]
                logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]
            self.log_dict({'step': self.current_epoch, **
                          {net_name+k: v for k, v in logs.items()}})

    def test_step(self, batch, batch_idx):
        # the test step is used to evaluate the quantized model
        x, files, y, device_indices, cities, indices = batch
        if self.random_sec:
            # x is of shape: batch_size, 1, 320000
            y = repeat(y, 'b -> (b 10)')
            x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
            files = repeat(np.array(files), 'b -> (b 10)')
            device_indices = repeat(device_indices, 'b -> (b 10)')
            cities = repeat(cities, 'b -> (b 10)')
            indices = repeat(indices, 'b -> (b 10)')
        x, y = x.cpu(), y.cpu()
        x = self.mel_forward(x)
        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        results = {"quant_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
        return results

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['quant_loss'] for x in outputs]).mean()
        quant_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'quant.loss': avg_loss, 'quant_acc': quant_acc, 'step': self.current_epoch}

        self.log_dict(logs)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f = batch
        if self.mel:
            x = self.mel_forward(x)
        if self.quantized_mode:
            x = x.cpu()
        y_hat = self.forward(x)
        return f, y_hat

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.parameters())
        return {
            'optimizer': optimizer,
            'lr_scheduler': get_lr_scheduler(optimizer)
        }

    def configure_callbacks(self):
        return get_callbacks()


@ex.command
def get_callbacks(swa=False, quantization=False,  swa_epoch_start=600,
                 swa_freq=3):
    callbacks = []
    if quantization:
        callbacks.append(MyStaticPostQuantizationCallback(get_train_loader))
    if swa:
        from helpers.swa_callback import StochasticWeightAveraging
        print("\n Using swa!\n")
        callbacks.append(StochasticWeightAveraging(
            swa_epoch_start=swa_epoch_start, swa_freq=swa_freq))
    return callbacks


def set_default_json_pickle(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


@ex.command
def check_load_compitable(preload, _config):
    DB_URL = preload['db_server'] or "mongodb://rk2:37373/?retryWrites=true&w=majority"
    DB_NAME = preload['db_name'] or "ba3l_dcase21t1a"
    import pymongo
    from pymongo import MongoClient
    mongodb_client = MongoClient(DB_URL)
    mongodb = mongodb_client[DB_NAME]
    e = mongodb["runs"].find_one({"_id": preload['run_id']})
    assert _config["models"]['net']["arch"]==e['config']["models"]['net']["arch"]


@ex.command(prefix="preload")
def get_pred_file(run_id=None, db_server=None, db_name=None, file_name=None):
    DB_URL = db_server or "mongodb://rk2:37373/?retryWrites=true&w=majority"
    DB_NAME = db_name or "ast_dcase22t1"
    import pymongo
    from pymongo import MongoClient
    mongodb_client = MongoClient(DB_URL)
    mongodb = mongodb_client[DB_NAME]
    e = mongodb["runs"].find_one({"_id": run_id})
    exp_name = e["experiment"]["name"]
    run_id = str(DB_NAME) + "_" + str(e['_id'])
    host_name = e['host']['hostname'].replace("rechenknecht", "rk").replace(".cp.jku.at", "")
    output_dir = "dcase22/malach_dcase22/" + e["config"]["trainer"]['default_root_dir']
    exp_path = f"/share/{host_name}/home/fschmid/deployment/{output_dir}/{exp_name}/{run_id}"
    assert os.path.isdir(exp_path)
    pred_path = f"{exp_path}/predictions"
    os.makedirs(pred_path, exist_ok=True)
    FILE_NAME = file_name or "default.pt"
    pred_file =f"{pred_path}/{FILE_NAME}"
    return pred_file


@ex.command
def evaluate(_run, _config, _log, _rnd, _seed, only_validation=True):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    check_load_compatible()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    net_statedict = get_net_state_dict()
    modul = M(ex)
    modul.net.load_state_dict(net_statedict)
    if not only_validation:
        print(f"\n\nNow testing training data, len{len(train_loader)}:")
        res = trainer.validate(modul, val_dataloaders=train_loader)
        print(res)
    modul.val_dataloader=None
    trainer.val_dataloaders = None
    print(f"\n\nValidation len={len(val_loader)}\n")
    res = trainer.validate(modul, val_dataloaders=val_loader)
    print("\n\n Validtaion:")
    print(res)


@ex.command(prefix="preload")
def get_net_state_dict(run_id=None, db_name=None):
    pl_ckpt = get_pl_ckpt(run_id=run_id, db_name=db_name)
    net_statedict = {k[4:]: v for k, v in pl_ckpt['state_dict'].items() if k.startswith("net.")}
    return net_statedict

@ex.command(prefix="teacher")
def get_teacher_state_dict(run_id=None, db_name="ast_dcase22t1"):
    pl_ckpt = get_pl_ckpt(run_id=run_id, db_name=db_name)
    net_statedict = {k[4:]: v for k, v in pl_ckpt['state_dict'].items() if k.startswith("net.")}
    return net_statedict


@ex.command(prefix="preload")
def get_pl_ckpt(ckpt=None, run_id=None, db_server=None, db_name=None):
    if ckpt is None:
        DB_URL = db_server or "mongodb://rk2:37373/?retryWrites=true&w=majority"
        DB_NAME = db_name or "ast_dcase22t1"
        import pymongo
        from pymongo import MongoClient
        mongodb_client = MongoClient(DB_URL)
        mongodb = mongodb_client[DB_NAME]
        e = mongodb["runs"].find_one({"_id": run_id})
        exp_name = e["experiment"]["name"]
        run_id = str(DB_NAME) + "_" + str(e['_id'])
        host_name = e['host']['hostname'].replace("rechenknecht", "rk").replace(".cp.jku.at", "")
        output_dir = "dcase22/malach_dcase22/" + e["config"]["trainer"]['default_root_dir']
        if "node" in host_name:
            ckpts_path = f"/share/rk6/shared/dcase22/khaled_models/{run_id}/checkpoints/"
        else:
            ckpts_path = f"/share/{host_name}/home/fschmid/deployment/{output_dir}/{exp_name}/{run_id}/checkpoints/"
        assert os.path.isdir(ckpts_path)
        ckpt = ckpts_path + os.listdir(ckpts_path)[-1]
    elif run_id is not None:
        print("\n\nWARNING: ckpt is given ignoring the run_id argument.\n\n")
    pl_ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
    return pl_ckpt


@ex.command
def check_load_compatible(preload, _config):
    DB_URL = preload['db_server'] or "mongodb://rk2:37373/?retryWrites=true&w=majority"
    DB_NAME = preload['db_name'] or "dcase22"
    import pymongo
    from pymongo import MongoClient
    mongodb_client = MongoClient(DB_URL)
    mongodb = mongodb_client[DB_NAME]
    e = mongodb["runs"].find_one({"_id": preload['run_id']})
    assert _config["models"]['net']["rho"] == e['config']["models"]['net']["rho"]
    assert _config["models"]['net']["groups_num"] == e['config']["models"]['net']["groups_num"]
    assert _config["models"]['net']["arch"] == e['config']["models"]['net']["arch"]
    assert _config["models"]['net']["cut_stage2"] == e['config']["models"]['net']["cut_stage2"]
    assert _config["models"]['net']["cut_stage3"] == e['config']["models"]['net']["cut_stage3"]


@ex.command
def main(_run, _config, _log, _rnd, seed):
    # seed = sacred root-seed, which is used to automatically seed random, numpy and pytorch
    seed_sequence = np.random.SeedSequence(seed)

    # seed torch, numpy, random with different seeds in main thread
    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)

    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()

    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    test_loader = ex.get_dataloaders(dict(test=True))

    sample = next(iter(train_loader))[0][0].unsqueeze(0)
    modul = M(ex)
    if modul.random_sec:
        sample = sample[:, :, :sample.size(2) // 10]

    shape = modul.mel_forward(sample).size()
    macc, n_params = nessi.get_model_size(modul.net, input_size=shape)
    ex.info['Spectrum Shape'] = shape
    ex.info['macc'] = macc
    ex.info['n_params'] = n_params

    trainer.fit(
            modul,
            train_dataloader=train_loader,
            val_dataloaders=val_loader,
        )
    # we only run test to test quantization accuracy
    if _config['quantization']:
        trainer.test(test_dataloaders=test_loader)
    return {"done": True}


@ex.automain
def default_command():
    return main()

