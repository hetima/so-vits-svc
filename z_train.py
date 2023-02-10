import os
import sys
import argparse
from InquirerPy import inquirer

PROJECT_PATH = "./projects"
PRETRAINED_MODELS_PATH = "./pretrained_models"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default="", help="project name")

    args = parser.parse_args()
    # processs = cpu_count()-2 if cpu_count() >4 else 1
    # pool = Pool(processes=processs)
    if args.project == "":
        # args.project = inquirer.text(message="input project name:", default="").execute()
        args.project = select_project()
    if args.project == "":
        print("project name is empty")
        exit(0)
    config_path = os.path.join(PROJECT_PATH, args.project, "config.json")
    sys.argv = ["", "-c", config_path, "-m", args.project]
    import train
    train.main()

    
