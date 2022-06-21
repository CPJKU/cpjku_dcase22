import scipy
from sacred.config_helpers import DynamicIngredient, CMD
import torch
import scipy.io.wavfile as wavfile
import os


LMODE = os.environ.get("LMODE", False)


def add_configs(ex):

    # dataset config for random cuts from 10 second audios.
    @ex.named_config
    def dcase22_reassembled():
        if LMODE:
            basedataset = DynamicIngredient("datasets.dcase22.dcase22t1.dataset",
                                            name="d22t1_r",
                                            base_dir="/system/user/publicdata/CP/DCASE/dcase22_reassembled/",
                                            audio_path="/system/user/publicdata/CP/DCASE/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/",
                                            meta_csv="/system/user/publicdata/CP/DCASE/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv",
                                            train_files_csv="/system/user/publicdata/CP/DCASE/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_train.csv",
                                            test_files_csv="/system/user/publicdata/CP/DCASE/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_evaluate.csv"
                                            )
        else:
            basedataset = DynamicIngredient("datasets.dcase22.dcase22t1.dataset",
                                            name="d22t1_r",
                                            base_dir="/share/rk6/shared/dcase22_reassembled/",
                                            audio_path="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/",
                                            meta_csv="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv",
                                            train_files_csv="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_train.csv",
                                            test_files_csv="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_evaluate.csv"
                                            )

    # config for cp_resnet designed for the limited number of parameters
    @ex.named_config
    def cp_mini_resnet():
        models = {
            "net": DynamicIngredient("models.cp.cp_resnet.model_ing", n_blocks=(2, 1, 1))
        }
        datasets = dict(training=dict(batch_size=64), validate=dict(batch_size=64))
        weight_decay = 1e-3
        lr = 0.001

    # extra commands
    @ex.command
    def test_ir():
        itr = ex.datasets.training.get_iter()
        import time
        start = time.time()
        print("hello")
        for i, b in enumerate(itr):
            for x,name, label in zip(*b):
                x=x.numpy()
                print(name,x.shape,label)
                wavfile.write(name.replace("audio/",""),22050,x[0])
            break
        end = time.time()
        print("totoal time:", end - start)

    @ex.command
    def test_loaders_speed():
        itr = ex.datasets.training.get_iter()
        import time
        start = time.time()
        print("hello")
        for i, b in enumerate(itr):
            if i % 20 == 0:
                print(f"{i}/{len(itr)}", end="\r")
        end = time.time()
        print("totoal time:", end - start)
        start = time.time()
        print("retry:")
        for i, b in enumerate(itr):
            if i % 20 == 0:
                print(f"{i}/{len(itr)}", end="\r")
        end = time.time()
        print("totoal time:", end - start)

    @ex.command
    def test_stft_speed():
        torch.backends.cudnn.benchmark = True
        itr = ex.datasets.training.get_iter()
        import time
        with torch.cuda.amp.autocast():
            print("float16")
            for _ in range(2):
                start = time.time()
                print("hello")
                for i, b in enumerate(itr):
                    if i % 20 == 0:
                        print(f"{i}/{len(itr)}", end="\r")
                end = time.time()
                print("Pure load time:", end - start)
                from models.preprocess.spectrograms import STFT
                nnaudio = STFT().cuda()
                from models.preprocess.spectrograms import JANCONVSTFT, JANSTFT,PerceptualMelSpectrogram
                janconv = JANCONVSTFT().cuda()
                janstft = JANSTFT().cuda()
                PMS = PerceptualMelSpectrogram().cuda()
                start = time.time()
                print("nnaudio")
                a = torch.as_tensor(0, device="cuda")
                for i, b in enumerate(itr):
                    if i % 20 == 0:
                        print(f"{i}/{len(itr)}", end="\r")
                    x = b[0].cuda()
                    old_shape = x.size()
                    x = x.reshape(-1, old_shape[2])
                    res = nnaudio(x)
                    if i == 0:
                        print()
                        print(res.shape, end="\n")
                        print()
                    a = a + res.abs().mean()
                print("res:", a.item())
                end = time.time()
                print("totoal nnaudio time:", end - start)
                print("JANCONVSTFT:")
                start = time.time()
                a = torch.as_tensor(0, device="cuda")
                for i, b in enumerate(itr):
                    if i % 20 == 0:
                        print(f"{i}/{len(itr)}", end="\r")
                    x = b[0].cuda()
                    old_shape = x.size()
                    x = x.reshape(-1, old_shape[2])
                    res = janconv(x)
                    if i == 0:
                        print()
                        print(res.shape, end="\n")
                        print()
                    a = a + res.abs().mean()
                print("res:", a.item())
                end = time.time()
                print("total JANCONVSTFT time:", end - start)
                print("JANSTFT:")
                start = time.time()
                a = torch.as_tensor(0, device="cuda")
                for i, b in enumerate(itr):
                    if i % 20 == 0:
                        print(f"{i}/{len(itr)}", end="\r")
                    x = b[0].cuda()
                    old_shape = x.size()
                    x = x.reshape(-1, old_shape[2])
                    res = janstft(x)
                    if i == 0:
                        print()
                        print(res.shape, end="\n")
                        print()
                    a = a + res.abs().mean()
                print("res:", a.item())
                end = time.time()
                print("total JANSTFT time:", end - start)
                print("PMS:")
                start = time.time()
                a = torch.as_tensor(0, device="cuda")
                for i, b in enumerate(itr):
                    if i % 20 == 0:
                        print(f"{i}/{len(itr)}", end="\r")
                    x = b[0].cuda()
                    old_shape = x.size()
                    x = x.reshape(-1, old_shape[2])
                    res = PMS(x)
                    if i == 0:
                        print()
                        print(res.shape, end="\n")
                        print()
                    a = a + res.abs().mean()
                print("res:", a.item())
                end = time.time()
                print("total PMS time:", end - start)

