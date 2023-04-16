# for 4.0

import os
import argparse
from InquirerPy import inquirer
from glob import iglob
from tqdm import tqdm
import json
import wave
import re
import torch
import utils
import librosa
import numpy as np

PROJECT_PATH = "./projects"

config_template = json.load(open("configs/config.json"))
# config_template = {
#   "train": {
#     "log_interval": 200,
#     "eval_interval": 800,
#     "seed": 1234,
#     "epochs": 10000,
#     "learning_rate": 0.0001,
#     "betas": [
#       0.8,
#       0.99
#     ],
#     "eps": 1e-09,
#     "batch_size": 6,
#     "fp16_run": false,
#     "lr_decay": 0.999875,
#     "segment_size": 10240,
#     "init_lr_ratio": 1,
#     "warmup_epochs": 0,
#     "c_mel": 45,
#     "c_kl": 1.0,
#     "use_sr": true,
#     "max_speclen": 512,
#     "port": "8001",
#     "keep_ckpts": 3
#   },
#   "data": {
#     "training_files": "filelists/train.txt",
#     "validation_files": "filelists/val.txt",
#     "max_wav_value": 32768.0,
#     "sampling_rate": 44100,
#     "filter_length": 2048,
#     "hop_length": 512,
#     "win_length": 2048,
#     "n_mel_channels": 80,
#     "mel_fmin": 0.0,
#     "mel_fmax": 22050
#   },
#   "model": {
#     "inter_channels": 192,
#     "hidden_channels": 192,
#     "filter_channels": 768,
#     "n_heads": 2,
#     "n_layers": 6,
#     "kernel_size": 3,
#     "p_dropout": 0.1,
#     "resblock": "1",
#     "resblock_kernel_sizes": [3,7,11],
#     "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
#     "upsample_rates": [ 8, 8, 2, 2, 2],
#     "upsample_initial_channel": 512,
#     "upsample_kernel_sizes": [16,16, 4, 4, 4],
#     "n_layers_q": 3,
#     "use_spectral_norm": false,
#     "gin_channels": 256,
#     "ssl_dim": 256,
#     "n_speakers": 200
#   },
#   "spk": {
#     "nyaru": 0,
#     "huiyu": 1,
#     "nen": 2,
#     "paimon": 3,
#     "yunhao": 4
#   }
# }

def select_project():
    dir_path = PROJECT_PATH
    dirs = []
    for f in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, f)):
            dirs.append(f)
    dirs.append("Cancel")
    slct = inquirer.rawlist(message="Select project:", choices=dirs, multiselect=False).execute()
    if slct == 'Cancel': slct = ''
    return slct

#convert wav
def convert_wav(item):
    spkdir, wav_path, wav_name, args = item
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    # wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, sr=None)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        wav2 /= max(wav2.max(), -wav2.min())
        save_name = wav_name
        save_path2 = os.path.join(args.out_dir2, speaker, save_name)
        wavfile.write(
            save_path2,
            args.sr2,
            (wav2 * np.iinfo(np.int16).max).astype(np.int16)
        )


# preprocess_flist_config.py
pattern = re.compile(r'^[\.a-zA-Z0-9_\/()]+$')
#make config and filelists
def preprocess_flist_config(args):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    # parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    # parser.add_argument("--test_list", type=str, default="./filelists/test.txt", help="path to test list")
    # parser.add_argument("--source_dir", type=str, default="./dataset/32k", help="path to source dir")
    # args = parser.parse_args()

    args.train_list = os.path.join(args.prj_dir, "filelists/train.txt").replace("\\", "/")
    args.val_list = os.path.join(args.prj_dir, "filelists/val.txt").replace("\\", "/")
    args.test_list = os.path.join(args.prj_dir, "filelists/test.txt").replace("\\", "/")
    args.source_dir = os.path.join(args.prj_dir, "dataset").replace("\\", "/")
    os.makedirs(os.path.join(args.prj_dir, "filelists"), exist_ok=True)
    train = []
    val = []
    test = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = ["/".join([args.source_dir, speaker, i]) for i in os.listdir(os.path.join(args.source_dir, speaker))]

        new_wavs = []
        for file in wavs:
            if not file.endswith("wav"):
                continue
            if not pattern.match(file):
                print(f"warning：文件名{file}中包含非字母数字下划线，可能会导致错误。（也可能不会）")
            if get_wav_duration(file) < 0.3:
                print("skip too short audio:", file)
                continue
            new_wavs.append(file)
        wavs = new_wavs
        shuffle(wavs)
        train += wavs[2:-2]
        val += wavs[:2]
        test += wavs[-2:]
    # n_speakers = len(spk_dict.keys()) * 2
    shuffle(train)
    shuffle(val)
    shuffle(test)

    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath.replace("\\", "/") + "\n")

    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath.replace("\\", "/") + "\n")

    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for fname in tqdm(test):
            wavpath = fname
            f.write(wavpath.replace("\\", "/") + "\n")

    # config_template["model"]["n_speakers"] = n_speakers
    config_template["spk"] = spk_dict
    config_template["data"]["training_files"] = args.train_list
    config_template["data"]["validation_files"] = args.val_list
    print("Writing configs/config.json")
    with open(os.path.join(args.prj_dir, "config.json"), "w") as f:
        json.dump(config_template, f, indent=2)


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # 获取音频帧数
        n_frames = wav_file.getnframes()
        # 获取采样率
        framerate = wav_file.getframerate()
        # 计算时长（秒）
        duration = n_frames / float(framerate)
    return duration

# preprocess_flist_config.py end

# preprocess_hubert_f0.py
sampling_rate = 0
hop_length = 0
def preprocess_hubert_f0(in_dir):
    import multiprocessing
    import math
    # print("Loading hubert for content...")
    # hmodel = utils.get_hubert_model(0 if torch.cuda.is_available() else None)
    # print("Loaded hubert.")
    # filenames = glob(f'{in_dir}/*/*.wav', recursive=True)  #[:10]
    # for filename in tqdm(filenames):
    #     process(filename, hmodel)
    filenames = glob(f'{in_dir}/*/*.wav', recursive=True)  # [:10] dataset/44k
    shuffle(filenames)
    multiprocessing.set_start_method('spawn',force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]
    print([len(c) for c in chunks])
    processes = [multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks]
    for p in processes:
        p.start()


def process_one(filename, hmodel):
    # 取ってくるのが面倒なので決め打ち
    sampling_rate = 44100
    hop_length = 512
    # print(sampling_rate)
    # print(filename)
    wav, sr = librosa.load(filename, sr=sampling_rate)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(devive)
        c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k)
        torch.save(c.cpu(), soft_path)
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0 = utils.compute_f0_dio(wav, sampling_rate=sampling_rate, hop_length=hop_length)
        np.save(f0_path, f0)

def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmodel = utils.get_hubert_model().to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, hmodel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default="", help="project name")
    parser.add_argument("--sr2", type=int, default=44100, help="sampling rate")
    # parser.add_argument("--in_dir", type=str, default="./dataset_raw", help="path to source dir")
    # parser.add_argument("--out_dir2", type=str, default="./dataset_raw_outttest", help="path to target dir")
    args = parser.parse_args()



    if args.project == "":
        # args.project = inquirer.text(message="input project name:", default="").execute()
        args.project = select_project()
    if args.project == "":
        print("project name is empty")
    else:
        import numpy as np
        from scipy.io import wavfile
        from random import shuffle
        import json

        from glob import glob
        from pyworld import pyworld
        from scipy.io import wavfile
        # from mel_processing import mel_spectrogram_torch
        import logging
        logging.getLogger('numba').setLevel(logging.WARNING)

        #resample
        args.prj_dir = os.path.join(PROJECT_PATH, args.project)
        args.out_dir2 = os.path.join(args.prj_dir, "dataset")
        args.in_dir = os.path.join(args.prj_dir, "raw")
        for speaker in os.listdir(args.in_dir):
            spk_dir = os.path.join(args.in_dir, speaker).replace("\\","/")

            if os.path.isdir(spk_dir):
                print(spk_dir)
                for i in iglob(os.path.join(spk_dir, "**/*.wav"), recursive=True):
                    wav_path = i.replace("\\", "/")
                    wav_name = wav_path.replace(spk_dir, "").replace("/", "_")
                    print(wav_name)
                    convert_wav((spk_dir, wav_path, wav_name, args))
        #preprocess_flist_config.py
        preprocess_flist_config(args)
        #preprocess_hubert_f0.py
        hps = utils.get_hparams_from_file(os.path.join(args.prj_dir, "config.json"))
        sampling_rate = hps.data.sampling_rate
        hop_length = hps.data.hop_length
        preprocess_hubert_f0(args.out_dir2)
