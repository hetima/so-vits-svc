import os
import shutil
import argparse
from InquirerPy import inquirer

PROJECT_PATH = "./projects"
PRETRAINED_MODELS_PATH = "./pretrained_models"

def init_project(prj_name):
    prj_dir = os.path.join(PROJECT_PATH, prj_name)
    log_dir = os.path.join("./logs", prj_name)  #os.path.join(prj_dir, "checkpoints")
    os.makedirs(os.path.join(prj_dir, "raw", prj_name), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    d_file = os.path.join(log_dir, "D_0.pth")
    g_file = os.path.join(log_dir, "G_0.pth")
    if not os.path.exists(d_file):
        shutil.copyfile(os.path.join(PRETRAINED_MODELS_PATH, "D_0.pth"), d_file)
    if not os.path.exists(g_file):
        shutil.copyfile(os.path.join(PRETRAINED_MODELS_PATH, "G_0.pth"), g_file)
    print(prj_name + " inited")
    print("put wav files into " + os.path.join(prj_dir, "raw", "*speaker_id*").replace("\\", "/"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default="", help="project name")

    args = parser.parse_args()
    # processs = cpu_count()-2 if cpu_count() >4 else 1
    # pool = Pool(processes=processs)

    if args.project == "":
        args.project = inquirer.text(message="input project name:", default="").execute()
    if args.project == "":
        print("project name is empty")
    else:
        init_project(args.project)
