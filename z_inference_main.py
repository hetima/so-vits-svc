# for 4.0
# 引数 --export_to_same_dir を付けると変換元のwavと同じフォルダに書き出す
# kmeans_10000.pt が存在したら cluster_infer_ratio を訊いてくる。0にすれば使用しない

import os
import re
import glob
import json
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator
import argparse

LOG_PATH = "./logs"


def z_latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if len(f_list) <= 0:
        return ""
    x = f_list[-1]
    return x

def select_project():
    dir_path = LOG_PATH
    dirs = []
    for f in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, f)):
            if os.path.exists(os.path.join(dir_path, f, "config.json")):
                dirs.append(f)
    dirs.append("Cancel")
    slct = inquirer.rawlist(message="Select project:", choices=dirs, multiselect=False).execute()
    if slct == 'Cancel': slct = ''
    return slct

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sovits4 inference')
    parser.add_argument('--export_to_same_dir', action='store_true', default=False, help='Export to the same directory as the input wav')
    # parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='聚类模型路径，如果没有训练聚类则随便填')
    # parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案占比，范围0-1，若没有训练聚类模型则填0即可')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    args = parser.parse_args()
    filename_label = ""

    #model
    project = select_project()
    if project == "":
        print("project name is empty")
        exit(0)
    model_path = z_latest_checkpoint_path(os.path.join(LOG_PATH, project), "G_*.pth")
    if model_path == "":
        print("model file not found")
        exit(0)
    print(os.path.basename(model_path))
    model_step = re.sub(r"\D", "", os.path.basename(model_path))
    #speaker
    config_path = os.path.join(LOG_PATH, project, "config.json")
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    spk_list = list(config["spk"].keys())
    if len(spk_list) == 1:
        spk = spk_list[0]
        print("spk = " + spk)
    else:
        spk = inquirer.rawlist(message="Select speaker:", choices=spk_list, multiselect=False).execute()
    #slice_db
    # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
    slice_db = inquirer.text(message="slice threshold db:", default="-40", validate=NumberValidator()).execute()
    slice_db = int(slice_db)

    #cluster_model_path
    cluster_infer_ratio = 0  # 0-1.0,
    cluster_model_path = z_latest_checkpoint_path(os.path.join(LOG_PATH, project), "kmeans_*.pt")
    if cluster_model_path != "":
        print("found cluster model:" + cluster_model_path)
        cluster_infer_ratio = inquirer.text(message="cluster infer ratio(0 - 1.0):", default="0").execute()
        filename_label = "c" + cluster_infer_ratio
        cluster_infer_ratio = float(cluster_infer_ratio)
    if cluster_infer_ratio <= 0:
        filename_label = ""

    #wav file
    src_path = inquirer.filepath(message="wav file path:").execute()
    if src_path[0] == '"' or src_path[0] == "'":
        src_path = src_path[1:-1]
    if not os.path.exists(src_path):
        print("file not found")
        exit(0)

    # other config
    tran = 0  # 音高调整，支持正负（半音）-5

    auto_predict_f0 = args.auto_predict_f0  # False  #语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
    noice_scale = args.noice_scale  #0.4  #噪音级别，会影响咬字和音质，较为玄学
    pad_seconds = 0.5  #推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现

    if auto_predict_f0:
        filename_label = filename_label + "a"

    import io
    import logging
    # import time
    # import librosa
    import numpy as np
    import soundfile
    from inference import infer_tool
    from inference import slicer
    from inference.infer_tool import Svc

    logging.getLogger('numba').setLevel(logging.WARNING)

    if cluster_model_path == '' or cluster_infer_ratio <= 0:
        svc_model = Svc(model_path, config_path, args.device, "")
    else:
        svc_model = Svc(model_path, config_path, args.device, cluster_model_path)
        print("cluster_model_path = " + cluster_model_path)
    infer_tool.mkdir(["results"])

    clean_name = os.path.splitext(os.path.basename(src_path))[0]
    wav_format = 'wav'  # 音频输出格式

    raw_audio_path = src_path

    #convert to wav
    # infer_tool.format_wav(raw_audio_path)

    wav_path = raw_audio_path #Path(raw_audio_path).with_suffix('.wav')
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)


    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))

        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * pad_seconds)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = svc_model.infer(spk, tran, raw_path,
                                                cluster_infer_ratio=cluster_infer_ratio,
                                                auto_predict_f0=auto_predict_f0,
                                                noice_scale=noice_scale
                                                )
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * pad_seconds)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))

    # res_path = f'./results/{clean_name}_{tran}key_{spk}.{wav_format}'
    if args.export_to_same_dir:
        parent_path = os.path.dirname(src_path)
        res_path = os.path.join(parent_path, f'{spk}_{model_step}{filename_label}_{clean_name}.{wav_format}')
    else:
        res_path = f'./results/{spk}_{model_step}{filename_label}_{clean_name}.{wav_format}'
    soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
