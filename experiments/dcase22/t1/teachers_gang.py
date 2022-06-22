from pathlib import Path

from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
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

# Creates new experiment
ex = Experiment("gang")

# define datasets and dataloaders
get_train_loader = ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True,
                                             batch_size=64, num_workers=8, shuffle=True,
                                             dataset=CMD("/basedataset.get_training_set_raw"))

get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=64, num_workers=8,
                                            dataset=CMD("/basedataset.get_test_set_raw"))

# prepared for evaluating fully trained model on test split of development set
ex.datasets.quantized_test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), test=True,
                                            batch_size=64, num_workers=8, dataset=CMD("/basedataset.get_test_set_raw"))

# evaluation data
get_eval_loader = ex.datasets.evaluate.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                           batch_size=64, num_workers=8, dataset=CMD("/basedataset.get_eval_set_raw"))


# Default config for the trainer
@ex.config
def default_conf():
    cmd = " ".join(sys.argv)
    process_id = os.getpid()
    models = {
        # student network (to be trained)
        "net": DynamicIngredient("models.cp.cp_resnet.model_ing"),
        # dynamically creating spectrograms for student
        # dataset implementation fetches (possibly cached) signals with a defined resampling rate (32k - matches
        # transformer pre-training)
        "mel": DynamicIngredient("models.passt.preprocess.model_ing",
                                         instance_cmd="AugmentMelSTFT",
                                         n_mels=256, sr=32000, win_length=2048, hopsize=744, n_fft=2048, freqm=0,
                                         timem=0,
                                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                                         fmax_aug_range=1),
        # the teacher ensemble (weights to be loaded)
        "teacher_net": DynamicIngredient("models.passt.passt.model_ing", instance_cmd="get_teacher_avg_ensemble",
                                         teachers_list=[]),
        # dynamically creating spectrograms for teacher (different stft config than student, needs to match pre-training
        # on Audioset)
        "teacher_mel": DynamicIngredient("models.passt.preprocess.model_ing",
                                 instance_cmd="AugmentMelSTFT",
                                 n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0,
                                 timem=0,
                                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                                 fmax_aug_range=1)
    }
    basedataset = DynamicIngredient(
        "datasets.dcase22.dcase22t1.dataset", audio_processor=dict(sr=32000, resample_only=True))
    trainer = dict(max_epochs=750, gpus=1,
                   weights_summary='full', benchmark=True)
    device_ids = {'a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6'}
    label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                 'street_pedestrian', 'street_traffic', 'tram']
    lr = 0.001


# register extra possible configs
add_configs(ex)

@ex.command
def get_scheduler_lambda(warm_up_len=100, ramp_down_start=250, ramp_down_len=400, last_lr_value=0.005,
                         schedule_mode="exp_lin"):
    """
    @param warm_up_len: number of epochs for the lr to reach its maximum value
    @param ramp_down_start: control the epoch where decline of the lr starts
    @param ramp_down_len: number of epochs it takes for the lr to descend
    @param last_lr_value: final value of lr as a percentage of the original lr
    @param schedule_mode: method of scheduling 'exp_lin' and 'cos_cyc' are available
    @return: configured lr scheduler
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

        # all the available devices in development set
        self.device_ids = self.config.device_ids
        self.label_ids = self.config.label_ids
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # read config and set mixup and mixstyle configurations
        self.mixup_alpha = self.config.get("mixup_alpha", False)
        self.mixstyle_p = self.config.get("mixstyle_p", 0.0)
        self.mixstyle_alpha = self.config.get("mixstyle_alpha", 0.1)

        # define knowledge distillation parameters
        self.temperature = self.config.get("temperature", 3)
        self.soft_targets_weight = self.config.get("soft_targets_weight", 50)  # weight loss for soft targets
        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none")  # KL Divergence loss for soft targets
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # define parameters for 10-second teacher
        self.soft_targets_weight_long = self.config.get("soft_targets_weight_long", 0)
        if self.soft_targets_weight_long:
            # fetch pre-computed teacher predictions - no need to create predictions on the fly
            pred_file = get_teacher_preds(self.config.get("teacher_long_run_id", 227))
            teacher_logits = torch.load(pred_file)
            self.teacher_soft_targets_long = self.log_softmax(teacher_logits.float() / self.temperature)

    # forward pass of the student
    def forward(self, x):
        return self.net(x)

    # forward pass of the teacher
    def teacher_forward(self, x):
        self.teacher_net.eval()
        return self.teacher_net(x)

    # calculates only the mel spectogram of the input for the student
    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    # calculates only the mel spectogram of the input for the teacher
    def teacher_mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.teacher_mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    # training step for pytorch lightning
    def training_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch

        # x is a 10-second file and is of shape: batch_size, 1, 320000
        # randomly pick 1 second from 10 seconds
        n_samples_per_second = x.size(2) // 10
        t_start = np.random.randint(0, x.size(2)-n_samples_per_second)
        # crop one second audio
        x = x[:, :, t_start:t_start+n_samples_per_second]

        # create teacher spectrogram
        x_teacher = self.teacher_mel_forward(x)
        # create student spectrogram
        x = self.mel_forward(x)
        batch_size = len(y)

        # apply mixstyle to student and teacher spectorgrams
        if self.mixstyle_p > 0:
            x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)
            x_teacher = mixstyle(x_teacher, self.mixstyle_p, self.mixstyle_alpha)

        if self.mixup_alpha:
            rn_indices, lam = mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))
            # applying same mixup config also to teacher spectrograms
            x_teacher = x_teacher * lam.reshape(batch_size, 1, 1, 1) + \
                        x_teacher[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

            y_hat = self.forward(x)

            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))
        else:
            y_hat = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        label_loss = samples_loss.mean()
        samples_loss = samples_loss.detach()
        devices = [d.rsplit("-", 1)[1][:-4] for d in files]
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == y).sum()

        # getting teacher predictions
        with torch.no_grad():
            # inference step using teacher ensemble
            y_hat_teacher, embed = self.teacher_forward(x_teacher)

        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_soft_teacher = self.log_softmax(y_hat_teacher / self.temperature)
            y_soft_student = self.log_softmax(y_hat / self.temperature)

        # distillation loss
        soft_targets_loss = self.kl_div_loss(y_soft_student, y_soft_teacher).mean()  # mean since reduction="none"

        if self.soft_targets_weight_long:
            # getting teacher predictions on full 10-second snippets
            y_soft_teacher_long = self.teacher_soft_targets_long[indices].to(y_hat.device)
            if self.mixup_alpha:
                # if mixup is applied to student, we also have to mix 10-second teacher predictions
                soft_targets_loss_long = self.kl_div_loss(y_soft_student, y_soft_teacher_long).mean(dim=1) \
                                         * lam.reshape(batch_size) + \
                                    self.kl_div_loss(y_soft_student, y_soft_teacher_long[rn_indices]).mean(dim=1) \
                                         * (1. - lam.reshape(batch_size))
                soft_targets_loss_long = soft_targets_loss_long.mean()
            else:
                # mean since reduction="none"
                soft_targets_loss_long = self.kl_div_loss(y_soft_student, y_soft_teacher_long).mean()
            loss = self.soft_targets_weight * soft_targets_loss + \
                     self.soft_targets_weight_long * soft_targets_loss_long + label_loss
        else:
            loss = self.soft_targets_weight * soft_targets_loss + label_loss

        # logging results for losses and accuracies
        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}

        results["teacher_loss_weighted"] = soft_targets_loss.detach() * self.soft_targets_weight
        results["label_loss"] = label_loss.detach()

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

    # logging the training results at the end of epoch
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)

        logs = {'train.loss': avg_loss, 'train_acc': train_acc, 'step': self.current_epoch}
        avg_label_loss = torch.stack([x['label_loss'] for x in outputs]).mean()
        avg_teacher_loss_weighted = torch.stack([x['teacher_loss_weighted'] for x in outputs]).mean()
        avg_teacher_loss_long_weighted = torch.stack([x['teacher_loss_long_weighted'] for x in outputs]).mean()
        logs['label_loss'] = avg_label_loss
        logs['teacher_loss_weighted'] = avg_teacher_loss_weighted
        logs['teacher_loss_long_weighted'] = avg_teacher_loss_long_weighted

        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            logs["tloss." + d] = dev_loss / dev_cnt
            logs["tcnt." + d] = dev_cnt
        self.log_dict(logs)

    # validation step for pytorch lightning
    def validation_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch

        # x has been reassembled to 10 seconds - split it to 1-second snippets for validation
        y = repeat(y, 'b -> (b 10)')
        x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
        files = repeat(np.array(files), 'b -> (b 10)')
        device_indices = repeat(device_indices, 'b -> (b 10)')
        cities = repeat(cities, 'b -> (b 10)')
        indices = repeat(indices, 'b -> (b 10)')

        # create spectrograms and predictions
        x = self.mel_forward(x)
        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        devices = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(y):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_pred_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        return results

    # logging necessary loss and accuracies based on device and label
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc, 'step': self.current_epoch}

        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            dev_corrct = torch.stack([x["devn_correct." + d] for x in outputs]).sum()
            logs["vloss." + d] = dev_loss / dev_cnt
            logs["vacc." + d] = dev_corrct / dev_cnt
            logs["vcnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = torch.stack([x["lblloss." + l] for x in outputs]).sum()
            lbl_cnt = torch.stack([x["lblcnt." + l] for x in outputs]).sum()
            lbl_corrct = torch.stack([x["lbln_correct." + l] for x in outputs]).sum()
            logs["vloss." + l] = lbl_loss / lbl_cnt
            logs["vacc." + l] = lbl_corrct / lbl_cnt
            logs["vcnt." + l] = lbl_cnt
        self.log_dict(logs)

    # final test of the network based on the given dataset (quantized version of the network)
    # model is quantized before testing using helpers/utils.py - MyStaticPostQuantizationCallback
    def test_step(self, batch, batch_idx):
        # the test step is used to evaluate the quantized model
        x, files, y, device_indices, cities, indices = batch
        # x is of shape: batch_size, 1, 320000
        y = repeat(y, 'b -> (b 10)')
        x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
        files = repeat(np.array(files), 'b -> (b 10)')
        device_indices = repeat(device_indices, 'b -> (b 10)')
        cities = repeat(cities, 'b -> (b 10)')
        indices = repeat(indices, 'b -> (b 10)')
        # quantized model runs only on cpu
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

    # logging quantized loss of the network at the end of the training
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['quant_loss'] for x in outputs]).mean()
        quant_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'quant.loss': avg_loss, 'quant_acc': quant_acc, 'step': self.current_epoch}

        self.log_dict(logs)

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
def get_callbacks(quantization=False):
    callbacks = []
    if quantization:
        # register quantization callback
        # uses train loader to fetch calibration samples
        callbacks.append(MyStaticPostQuantizationCallback(get_train_loader))
    return callbacks


def get_teacher_preds(id_):
    return os.path.join("teacher_models", f"preds_{id_}.pt")


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

    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    test_loader = ex.get_dataloaders(dict(test=True))

    # get macc and n_parameters and log them as info
    # Attention: this is before fusing layers in the context of quantization - Parameters will slightly decrease and
    # MACs will slightly increase
    sample = next(iter(train_loader))[0][0].unsqueeze(0)
    modul = M(ex)
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

    # final validation results
    val_res = trainer.validate(val_dataloaders=val_loader)

    # if quantization=1 is set, we run testing of the quantized model
    if _config['quantization']:
        quant_res = trainer.test(test_dataloaders=test_loader)[0]
        quant_res = {"quant_loss": quant_res['quant.loss'], "quant_acc": quant_res['quant_acc']}
    else:
        quant_res = "Run with quantization=1 to obtain quantized results."

    print("**********************************")
    print("Final Validation Results: ")
    print("**********************************")
    print(val_res)

    print("\n\n")

    print("**********************************")
    print("Final Quantization Results: ")
    print("**********************************")
    print(quant_res)
    return {"done": True}


@ex.automain
def default_command():
    return main()

