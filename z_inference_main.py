import os
import re
import glob
import json
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator

LOG_PATH = "./logs"


def z_latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
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

    #model
    project = select_project()
    if project == "":
        print("project name is empty")
        exit(0)
    model_path = z_latest_checkpoint_path(os.path.join(LOG_PATH, project), "G_*.pth")
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
    #wav file
    src_path = inquirer.filepath(message="wav file path:").execute()
    if src_path[0] == '"' or src_path[0] == "'":
        src_path = src_path[1:-1]
    if not os.path.exists(src_path):
        print("file not found")
        exit(0)



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

    svc_model = Svc(model_path, config_path)
    infer_tool.mkdir(["results"])

    # 支持多个wav文件，放在raw文件夹下
    clean_name = os.path.splitext(os.path.basename(src_path))[0]
    wav_format = 'wav'  # 音频输出格式

    tran = 0  # 音高调整，支持正负（半音）-5
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
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
            _audio = out_audio.cpu().numpy()
        audio.extend(list(_audio))

    # res_path = f'./results/{clean_name}_{tran}key_{spk}.{wav_format}'
    res_path = f'./results/{spk}_{model_step}_{clean_name}.{wav_format}'
    soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
