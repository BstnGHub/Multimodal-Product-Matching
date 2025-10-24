import os
import shutil
import sys
from dataclasses import dataclass

from filelock import FileLock, Timeout
from configs.config import add_last_run_id_to_logfile
from typing import List


@dataclass
class TrainedModel:
    id: int         # run-id
    model_type: int # 0 = text, 1 = image, 2 = fusion
    value: float    # F1-Score
    location: str   # location of the model on the disk

current_trained_model = TrainedModel(
    id=0,
    model_type=-1,
    value=0,
    location="/placeholder"
)

top_text_models: List[TrainedModel] = []
top_image_models: List[TrainedModel] = []
top_fusion_models: List[TrainedModel] = []

def print_trained_models(top_k_models):
    """
    Print the trained models
    """

    print("Neue Rangliste: ")
    for model in top_k_models:
        print(model)

def create_model_dir_placeholder(config):
    """
    Creates the model directory placeholder
    :param config: json config
    """

    # Create storage location for the model
    # If a folder with the same ID already exists, select the next higher one

    lock = FileLock("locks/create_model_dir_placeholder.lock", timeout=5)

    try:
        with lock:

            dir_prefix_id = "0000"
            highest_run_id_atm = config.run_id
            highest_dir_numbers = []

            # The system should continue with the highest prefix. This prevents existing runs that are not found in the current directory
            # from being recreated. Prevents duplication.
            # The run_id from the config file is always trusted. Unless there is already a directory with a
            # higher id. In this case, the run.id in the config is also adjusted. It then receives the id of the latest run.

            for dir_name in os.listdir(config.output_folder):
                my_path = os.path.join(config.output_folder, dir_name)
                if os.path.isdir(my_path):
                    prefix = dir_name[:4]

                    if prefix.isdigit():
                        local_dir_id = int(prefix)

                        if local_dir_id > highest_run_id_atm:
                            # locally larger than in config. config is incorrect. Adjust and add.
                            # But it could still find a directory that is larger. e.g.
                            # The system has reached 9, but there are still 10. So no break.
                            highest_run_id_atm = local_dir_id
                            highest_dir_numbers.append(local_dir_id)


                        # ========================================================================================
                        elif local_dir_id == highest_run_id_atm:
                            # The IDs match and the locale is no larger than specified in the config.
                            # This means we have reached the most recent directory and will later write +1 to the prefix.
                            highest_run_id_atm = local_dir_id
                            highest_dir_numbers.append(local_dir_id)


                        elif local_dir_id < config.run_id:
                            # Found a smaller local file. The run_config.json is trusted.
                            # Possible desync between different systems. Or folders were deleted.
                            # Solution: Since other files are still being scanned, set highest to the one from run_config.json.
                            # Consequently, case 2 will be triggered during the next run or, if there was never a folder created from run_config.json,
                            # no break, as it cannot be guaranteed that the file system reads chronologically.
                            highest_dir_numbers.append(config.run_id)

            # No folder existed previously, potential desync or init. The run_config.json is believed.
            if len(highest_dir_numbers) == 0:
                dir_id = -1
            else:
                # Grab the largest element, which is also the highest entry.
                highest_dir_numbers.sort(reverse=True)
                dir_id = highest_dir_numbers[0]

            # + 1 because new run
            dir_id = dir_id + 1

            if dir_id < 10:
                # 001 - 009
                dir_prefix_id = "000" + str(dir_id)
            elif dir_id < 100 and dir_id > 9:
                # 010 - 099
                dir_prefix_id = "00" + str(dir_id)
            elif dir_id <= 999 and dir_id > 99:
                # 100 - 999
                dir_prefix_id = "0" + str(dir_id)
            elif dir_id <= 9999 and dir_id > 999:
                # 1000 - 9999
                dir_prefix_id = str(dir_id)
            elif dir_id == 9999 or dir_id < 0:
                raise ValueError("You have too many logs. Max Number is 999 and Min is 0") from None

            config.run_id = dir_id  # id now assigned,

            # Create the dir:
            os.makedirs(
                config.output_folder + dir_prefix_id)

            highest_dir_numbers.clear()

    except Timeout:
        print("[ERROR] Timeout in logging_tools.py: create_model_dir_placeholder. Create placeholder dir's failed.")

    return dir_prefix_id


def create_and_model_dir_and_console_logfile(dir_prefix_id, path_to_my_output, model_prefix ,config):
    """
    Creates the model directory and console logfile
    :param dir_prefix_id: directory prefix
    :param path_to_my_output: path to my output folder
    :param model_prefix: model prefix
    :param config: json config
    """

    lock = FileLock("locks/create_and_model_dir_and_console_logfile.lock", timeout=5)

    try:
        with lock:

            found_my_dir = False
            model_dir_to_be_renamed = ""

            for dir_name in os.listdir(config.output_folder):
                my_path = os.path.join(config.output_folder, dir_name)
                if os.path.isdir(my_path):
                    prefix = dir_name[:4]

                    if prefix.isdigit():
                        local_dir_id = int(prefix)

                        if local_dir_id == int(dir_prefix_id):
                            # Folder for the model found.
                            found_my_dir = True
                            model_dir_to_be_renamed = dir_prefix_id
                            break

            if found_my_dir is False:
                raise RuntimeError(
                    f"[ERROR] Could not find a directory named {dir_prefix_id}, which should be reserved for this run. Exiting.")

            target = config.output_folder + model_dir_to_be_renamed
            destination = config.output_folder + model_dir_to_be_renamed + "_" + model_prefix

            os.rename(target, destination)
            path_to_my_output = destination


            consol_logfile = open(path_to_my_output + "/console-logs_" + dir_prefix_id + ".txt", "x")

            if config.system != "local":

                sys.stdout = consol_logfile
                sys.stderr = consol_logfile

            else:
                print(
                    "NO LOGGING IN LOGGING.TXT. In you're run_config.json config.system = local. Logging in this mode was only in the console.")


    except Timeout:
        print("[ERROR] Timeout in logging_tools.py: create_and_model_dir_and_console_logfile. Find and rename dir failed.")

    return consol_logfile


def create_run_config_json_copy(path_to_my_output, dir_prefix_id):
    """
    Creates the run config json file
    :param path_to_my_output: path to my output folder
    :param dir_prefix_id: directory prefix
    """

    lock = FileLock("locks/create_run_config_json_copy.lock", timeout=5)

    try:
        with lock:

            if not os.path.exists(path_to_my_output):
                os.mkdir(path_to_my_output)

            config_filename = f"config_{dir_prefix_id}.json"
            path_to_jsonconfig = os.path.join("programcode/configs", "run_config.json")
            path_target = os.path.join(path_to_my_output, config_filename)

            os.makedirs(os.path.dirname(path_target), exist_ok=True)
            shutil.copy(path_to_jsonconfig, path_target)

            path_to_model_config_json = path_to_my_output + "/" + config_filename

    except Timeout:
        print("[ERROR] Timeout in logging_tools.py: create_run_config_json_copy. Loading config file failed.")

    return path_to_model_config_json


def check_image_for_imagemodel(config):
    """
    Changes image resolution, if needed.
    """

    config.image_height = 224
    config.image_width = 224

    print(f"[INFO] Image height: {config.image_height}, image width: {config.image_width}")
