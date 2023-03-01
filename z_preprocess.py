import os
import argparse
from InquirerPy import inquirer
from glob import iglob
from tqdm import tqdm





PROJECT_PATH = "./projects"
config_template = {
    "train": {
        "log_interval": 200,
        "eval_interval": 1000,
        "seed": 1234,
        "epochs": 10000,
        "learning_rate": 1e-4,
        "betas": [0.8, 0.99],
        "eps": 1e-9,
        "batch_size": 10, #12
        "fp16_run": False,
        "lr_decay": 0.999875,
        "segment_size": 17920,
        "init_lr_ratio": 1,
        "warmup_epochs": 0,
        "c_mel": 45,
        "c_kl": 1.0,
        "use_sr": True,
        "max_speclen": 384,
        "port": "8001"
    },
    "data": {
        "training_files": "filelists/train.txt",
        "validation_files": "filelists/val.txt",
        "max_wav_value": 32768.0,
        "sampling_rate": 32000,
        "filter_length": 1280,
        "hop_length": 320,
        "win_length": 1280,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": None
    },
    "model": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [10, 8, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "gin_channels": 256,
        "ssl_dim": 256,
        "n_speakers": 0,
    },
    "spk": {
        "nen": 0,
        "paimon": 1,
        "yunhao": 2
    }
}

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
        # for wavpath in wavs:
        #     if not pattern.match(wavpath):
        #         print(f"warning：文件名{wavpath}中包含非字母数字下划线，可能会导致错误。（也可能不会）")
        if len(wavs) < 10:
            print(f"warning: {speaker} dataset less than 10")
        wavs = [i for i in wavs if i.endswith("wav")]
        shuffle(wavs)
        train += wavs[2:-2]
        val += wavs[:2]
        test += wavs[-2:]
    n_speakers = len(spk_dict.keys()) * 2
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

    config_template["model"]["n_speakers"] = n_speakers
    config_template["spk"] = spk_dict
    config_template["data"]["training_files"] = args.train_list
    config_template["data"]["validation_files"] = args.val_list
    print("Writing configs/config.json")
    with open(os.path.join(args.prj_dir, "config.json"), "w") as f:
        json.dump(config_template, f, indent=2)




def get_f0(path, p_len=None, f0_up_key=0):
    x, _ = librosa.load(path, 32000)
    if p_len is None:
        p_len = x.shape[0] // 320
    else:
        assert abs(p_len - x.shape[0] // 320) < 3, (path, p_len, x.shape)
    time_step = 320 / 32000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 32000).to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size = (p_len - len(f0) + 1) // 2
    if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    f0bak = f0.copy()
    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0bak


def resize2d(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res


def compute_f0(path, c_len):
    x, sr = librosa.load(path, sr=32000)
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * 320 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, 32000)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    assert abs(c_len - x.shape[0] // 320) < 3, (c_len, f0.shape)

    return None, resize2d(f0, c_len)


def process(filename, hmodel):
    print(filename)
    save_name = filename + ".soft.pt"
    if not os.path.exists(save_name):
        devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav, _ = librosa.load(filename, sr=16000)
        wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
        c = utils.get_hubert_content(hmodel, wav)
        torch.save(c.cpu(), save_name)
    else:
        c = torch.load(save_name)
    f0path = filename + ".f0.npy"
    if not os.path.exists(f0path):
        cf0, f0 = compute_f0(filename, c.shape[-1] * 2)
        np.save(f0path, f0)


def preprocess_hubert_f0(in_dir):
    print("Loading hubert for content...")
    hmodel = utils.get_hubert_model(0 if torch.cuda.is_available() else None)
    print("Loaded hubert.")
    filenames = glob(f'{in_dir}/*/*.wav', recursive=True)  #[:10]
    for filename in tqdm(filenames):
        process(filename, hmodel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default="", help="project name")
    parser.add_argument("--sr2", type=int, default=32000, help="sampling rate")
    # parser.add_argument("--in_dir", type=str, default="./dataset_raw", help="path to source dir")
    # parser.add_argument("--out_dir2", type=str, default="./dataset_raw_outttest", help="path to target dir")
    args = parser.parse_args()



    if args.project == "":
        # args.project = inquirer.text(message="input project name:", default="").execute()
        args.project = select_project()
    if args.project == "":
        print("project name is empty")
    else:
        import librosa
        import numpy as np
        from scipy.io import wavfile
        from random import shuffle
        import json

        import torch
        from glob import glob
        from pyworld import pyworld
        from scipy.io import wavfile
        import utils
        from mel_processing import mel_spectrogram_torch
        import logging
        logging.getLogger('numba').setLevel(logging.WARNING)
        import parselmouth
        import numpy as np

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
        #make config and filelists
        preprocess_flist_config(args)
        #preprocess_hubert_f0
        preprocess_hubert_f0(args.out_dir2)
