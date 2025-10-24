from typing import Optional
import datetime
import json, os
from filelock import FileLock, Timeout
import numpy as np

class Config:

    # not in run_config.json:
    last_epoch_nr_before_reset: Optional[int] = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    model_id: Optional[str] = None

    # in run_config.json:
    run_id: Optional[int] = None
    start_date: None
    start_date_formated: None

    text_training_time: Optional[float] = None
    text_second_training_time: Optional[float] = None
    image_training_time: Optional[float] = None
    fusion_training_time: Optional[float] = None

    text_eval_time: Optional[float] = None
    image_eval_time: Optional[float] = None
    fusion_eval_time: Optional[float] = None

    end_date: None
    end_date_formated: None
    epochs: Optional[int] = None
    text_embedding_id: Optional[str] = None
    image_embedding_id: Optional[str] = None

    gpu: Optional[str] = None
    system: Optional[str] = None
    training_verbose: Optional[int] = None
    data_id: Optional[str] = None

    parallel_run: Optional[int] = None
    output_folder: Optional[str] = None
    path_to_load_model_from: Optional[str] = None
    dir_prefix_id: Optional[str] = None

    early_stopping_monitor: Optional[str] = None
    early_stopping_mode: Optional[str] = None
    early_stopping_patience: Optional[int] = None

    learning_rate: Optional[float] = None
    learning_rate_unfreeze: Optional[float] = None
    epoch_where_second_lr_started: Optional[int] = None
    batch_size: Optional[int] = None

    image_learning_rate: Optional[float] = None
    image_optimizer: Optional[str] = None
    text_optimizer: Optional[str] = None
    text_model_activate_second_training: Optional[bool] = None
    only_compile_text_model_learning_rate: Optional[float] = None

    text_architecture_bilstm_units: Optional[int] = None
    text_architecture_dropout1: Optional[float] = None
    text_architecture_dense1_units: Optional[int] = None
    text_architecture_dense1_activation: Optional[str] = None
    text_architecture_dropout2: Optional[float] = None
    text_architecture_dense2_units: Optional[int] = None
    text_architecture_dense2_activation: Optional[str] = None
    text_architecture_dropout3: Optional[float] = None
    text_architecture_output_activation: Optional[str] = None

    image_architecture_dropout: Optional[float] = None
    image_architecture_dense1_units: Optional[int] = None
    image_architecture_dense1_activation: Optional[str] = None
    image_architecture_dense2_units: Optional[int] = None
    image_architecture_dense2_activation: Optional[str] = None
    image_architecture_sister_output_dense: Optional[int] = None

    fusion_architecture_dropout1: Optional[float] = None
    fusion_architecture_dense1_units: Optional[int] = None
    fusion_architecture_dense1_activation: Optional[str] = None
    fusion_architecture_dropout2: Optional[float] = None
    fusion_architecture_dense2_units: Optional[int] = None
    fusion_architecture_dense2_activation: Optional[str] = None
    fusion_architecture_dropout3: Optional[float] = None
    fusion_architecture_dense3_units: Optional[int] = None
    fusion_architecture_dense3_activation: Optional[str] = None
    fusion_architecture_dropout4: Optional[float] = None
    fusion_architecture_output_activation: Optional[str] = None

    deleted_for_top_x: Optional[bool] = None

    text_loss: Optional[float] = None
    text_accuracy: Optional[float] = None
    text_precision: Optional[float] = None
    text_recall: Optional[float] = None
    text_auc: Optional[float] = None
    text_f1: Optional[float] = None

    image_loss: Optional[float] = None
    image_accuracy: Optional[float] = None
    image_precision: Optional[float] = None
    image_recall: Optional[float] = None
    image_auc: Optional[float] = None
    image_f1: Optional[float] = None

    fusion_loss: Optional[float] = None
    fusion_accuracy: Optional[float] = None
    fusion_precision: Optional[float] = None
    fusion_recall: Optional[float] = None
    fusion_auc: Optional[float] = None
    fusion_f1: Optional[float] = None

    text_training_start_time: None
    text_training_end_time: None
    image_training_start_time: None
    image_training_end_time: None
    fusion_training_start_time: None
    fusion_training_end_time: None

    text_eval_start_time: None
    text_eval_end_time: None
    image_eval_start_time: None
    image_eval_end_time: None
    fusion_eval_start_time: None
    fusion_eval_end_time: None

    text_second_training_start_time: None
    text_second_training_end_time: None

    def __init__(self, config_path):

        self.config_path: str = config_path
        self._load_config()
        self.image_height = 224
        self.image_width = 224


    def _load_config(self):
        """
        Loads the config file
        """

        lock = FileLock("locks/init_load_config.lock", timeout=5)

        try:
            with lock:

                if not os.path.exists(self.config_path):
                    raise FileNotFoundError(f"{self.config_path} not found.")
                with open(self.config_path, "r") as file:
                    config_data = json.load(file)
                for key, value in config_data.items():
                    setattr(self, key, value)
                    globals()[key] = value

        except Timeout:
            print("[ERROR] Timeout in config.py: _load_config. Loading config file failed.")


def formate_delta(delta):
    """
    formate the delta between timestamps
    """

    if hasattr(delta, "total_seconds"):
        total_seconds = int(delta.total_seconds())
    else:
        total_seconds = int(delta)
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    # transformation zu mm.ss
    new_delta = float(f"{minutes}.{seconds:02d}")
    return new_delta

def calculate_time_differences(config):
    """
    Calculate the time differences between two timestamps
    """

    # Training-times
    delta = config.text_training_end_time - config.text_training_start_time
    config.text_training_time = formate_delta(delta)

    delta = config.image_training_end_time - config.image_training_start_time
    config.image_training_time = formate_delta(delta)

    delta = config.fusion_training_end_time - config.fusion_training_start_time
    config.fusion_training_time = formate_delta(delta)

    if config.text_model_activate_second_training:
        delta = config.text_second_training_end_time - config.text_second_training_start_time
        config.text_second_training_time = formate_delta(delta)


    # Evaluation-times
    delta = config.text_eval_end_time - config.text_eval_start_time
    config.text_eval_time = formate_delta(delta)

    delta = config.image_eval_end_time - config.image_eval_start_time
    config.image_eval_time = formate_delta(delta)

    delta = config.fusion_eval_end_time - config.fusion_eval_start_time
    config.fusion_eval_time = formate_delta(delta)


def _f1_or_zero(val):
    """
    Puts the f1-score or a 0 as correct placeholder into the config file
    """

    # If there is nothing there, return 0.
    if val is None:
        return 0
    # If it is an ndarray, take the element
    if isinstance(val, np.ndarray):
        return val.item()
    # If float (or np.float32) is
    try:
        return float(val)
    except:
        # Fallback
        return 0


def write_config_changes_into_json_file(path_to_model_config_json, adjustments_before_copy, config):
    """
    Write configs to json file
    """

    lock = FileLock("locks/write_config_changes_into_json_file.lock", timeout=5)

    try:
        with lock:
            print("[INFO] Write config changes into JSON file")

            if adjustments_before_copy:

                # Function is used for writing back
                pass

            else:
                # Set end time
                config.end_date = datetime.datetime.now()
                config.end_date_formated = config.end_date.strftime("%d-%m-%Y_%H:%M:%S")

                # Determine datetime differences
                calculate_time_differences(config)


            with open(path_to_model_config_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["run_id"] = config.run_id
            data["start_date"] = str(config.start_date)
            data["start_date_formated"] = str(config.start_date_formated)

            data["text_training_time"] = config.text_training_time
            data["text_second_training_time"] = config.text_second_training_time
            data["image_training_time"] = config.image_training_time
            data["fusion_training_time"] = config.fusion_training_time

            data["text_eval_time"] = config.text_eval_time
            data["image_eval_time"] = config.image_eval_time
            data["fusion_eval_time"] = config.fusion_eval_time

            data["end_date"] = str(config.end_date)
            data["end_date_formated"] = str(config.end_date_formated)
            data["epochs"] = config.epochs
            data["text_embedding_id"] = config.text_embedding_id

            data["gpu"] = config.gpu
            data["system"] = config.system
            data["training_verbose"] = config.training_verbose
            data["data_id"] = config.data_id

            data["parallel_run"] = config.parallel_run
            data["output_folder"] = config.output_folder
            data["path_to_load_model_from"] = config.path_to_load_model_from
            data["dir_prefix_id"] = config.dir_prefix_id

            data["early_stopping_monitor"] = config.early_stopping_monitor
            data["early_stopping_mode"] = config.early_stopping_mode
            data["early_stopping_patience"] = config.early_stopping_patience

            data["learning_rate"] = config.learning_rate
            data["learning_rate_unfreeze"] = config.learning_rate_unfreeze
            data["epoch_where_second_lr_started"] = config.epoch_where_second_lr_started
            data["batch_size"] = config.batch_size

            data["image_learning_rate"] = config.image_learning_rate
            data["image_optimizer"] = config.image_optimizer
            data["text_optimizer"] = config.text_optimizer
            data["text_model_activate_second_training"] = config.text_model_activate_second_training
            data["only_compile_text_model_learning_rate"] = config.only_compile_text_model_learning_rate

            data["text_architecture_bilstm_units"] = config.text_architecture_bilstm_units
            data["text_architecture_dropout1"] = config.text_architecture_dropout1
            data["text_architecture_dense1_units"] = config.text_architecture_dense1_units
            data["text_architecture_dense1_activation"] = config.text_architecture_dense1_activation
            data["text_architecture_dropout2"] = config.text_architecture_dropout2
            data["text_architecture_dense2_units"] = config.text_architecture_dense2_units
            data["text_architecture_dense2_activation"] = config.text_architecture_dense2_activation
            data["text_architecture_dropout3"] = config.text_architecture_dropout3
            data["text_architecture_output_activation"] = config.text_architecture_output_activation

            data["image_architecture_dropout"] = config.image_architecture_dropout
            data["image_architecture_dense1_units"] = config.image_architecture_dense1_units
            data["image_architecture_dense1_activation"] = config.image_architecture_dense1_activation
            data["image_architecture_dense2_units"] = config.image_architecture_dense2_units
            data["image_architecture_dense2_activation"] = config.image_architecture_dense2_activation
            data["image_architecture_sister_output_dense"] = config.image_architecture_sister_output_dense

            data["fusion_architecture_dropout1"] = config.fusion_architecture_dropout1
            data["fusion_architecture_dense1_units"] = config.fusion_architecture_dense1_units
            data["fusion_architecture_dense1_activation"] = config.fusion_architecture_dense1_activation
            data["fusion_architecture_dropout2"] = config.fusion_architecture_dropout2
            data["fusion_architecture_dense2_units"] = config.fusion_architecture_dense2_units
            data["fusion_architecture_dense2_activation"] = config.fusion_architecture_dense2_activation
            data["fusion_architecture_dropout3"] = config.fusion_architecture_dropout3
            data["fusion_architecture_dense3_units"] = config.fusion_architecture_dense3_units
            data["fusion_architecture_dense3_activation"] = config.fusion_architecture_dense3_activation
            data["fusion_architecture_dropout4"] = config.fusion_architecture_dropout4
            data["fusion_architecture_output_activation"] = config.fusion_architecture_output_activation

            data["text_loss"] = config.text_loss
            data["text_accuracy"] = config.text_accuracy
            data["text_precision"] = config.text_precision
            data["text_recall"] = config.text_recall
            data["text_auc"] = config.text_auc
            data["text_f1"] = data["text_f1"] =_f1_or_zero(config.text_f1)

            data["image_loss"] = config.image_loss
            data["image_accuracy"] = config.image_accuracy
            data["image_precision"] = config.image_precision
            data["image_recall"] = config.image_recall
            data["image_auc"] = config.image_auc
            data["image_f1"] = data["image_f1"] = _f1_or_zero(config.image_f1)

            data["fusion_loss"] = config.fusion_loss
            data["fusion_accuracy"] = config.fusion_accuracy
            data["fusion_precision"] = config.fusion_precision
            data["fusion_recall"] = config.fusion_recall
            data["fusion_auc"] = config.fusion_auc
            data["fusion_f1"] = data["fusion_f1"] = _f1_or_zero(config.fusion_f1)

            data["text_training_start_time"] = str(config.text_training_start_time)
            data["text_training_end_time"] = str(config.text_training_end_time)
            data["image_training_start_time"] = str(config.image_training_start_time)
            data["image_training_end_time"] = str(config.image_training_end_time)
            data["fusion_training_start_time"] = str(config.fusion_training_start_time)
            data["fusion_training_end_time"] = str(config.fusion_training_end_time)

            data["text_eval_start_time"] = str(config.text_eval_start_time)
            data["text_eval_end_time"] = str(config.text_eval_end_time)
            data["image_eval_start_time"] = str(config.image_eval_start_time)
            data["image_eval_end_time"] = str(config.image_eval_end_time)
            data["fusion_eval_start_time"] = str(config.fusion_eval_start_time)
            data["fusion_eval_end_time"] = str(config.fusion_eval_end_time)

            data["text_second_training_start_time"] = str(config.text_second_training_start_time)
            data["text_second_training_end_time"] = str(config.text_second_training_end_time)

            with open(path_to_model_config_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    except Timeout:
        print("[ERROR] Timeout in config.py: write_config_changes_into_json_file. Loading config file failed.")

    print("[INFO] Finished: Write config changes into JSON file")


def log_evaluate_metrics(model, model_number, config):
    """
    Log evaluation metrics
    """

    print("[INFO] Logging evaluation metrics")

    if model_number == 1:
        config.text_loss = model[0]
        config.text_accuracy = model[1]
        config.text_precision = model[2]
        config.text_recall = model[3]
        config.text_auc = model[4]
        config.text_f1 = model[5]

    elif model_number == 2:
        config.image_loss = model[0]
        config.image_accuracy = model[1]
        config.image_precision = model[2]
        config.image_recall = model[3]
        config.image_auc = model[4]
        config.image_f1 = model[5]

    elif model_number == 3:
        config.fusion_loss = model[0]
        config.fusion_accuracy = model[1]
        config.fusion_precision = model[2]
        config.fusion_recall = model[3]
        config.fusion_auc = model[4]
        config.fusion_f1 = model[5]


def change_config_values_by_key(key, new_value, path_to_model_config_json):
    """
    Change config values by key
    """

    lock = FileLock("locks/change_config_values_by_key.lock", timeout=5)

    try:
        with lock:

            with open(path_to_model_config_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            # change
            data[key] = new_value

            # save
            with open(path_to_model_config_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    except Timeout:
        print("[ERROR] Timeout in config.py: change_config_values_by_key. Loading config file failed.")


def add_last_run_id_to_logfile(path_to_model_config_json, last_id):
    """
    Adds the current run ID to the default log file, so that the next run knows that the run ID was already used
    :param path_to_model_config_json: path to config json file
    :param last_id: last run ID
    :return:
    """

    lock = FileLock("locks/add_last_run_id_to_logfile.lock", timeout=5)

    try:
        with lock:
            # open
            with open(path_to_model_config_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            # change
            data["run_id"] = last_id

            # save
            with open(path_to_model_config_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    except Timeout:
        print("[ERROR] Timeout in config.py: add_last_run_id_to_logfile. Loading config file failed.")
