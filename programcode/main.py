#!/usr/bin/env python3
import datetime
import gc
import sys

import tensorflow as tf
from branches import data_preparation
from branches import text_model
from branches import image_model
from branches import fusion_model
from utils.logging_tools import *
from configs.config import *

current_trained_model = None

def run_text_branch(config):
    """
    Runs the text branch
    :param config: json config
    """

    if config.text_embedding_id == "multilingual-e5-large-instruct":
        config.model_id = "Fine-tuned-query-instruct"
    elif config.text_embedding_id == "multilingual-e5-small" or config.text_embedding_id == "multilingual-e5-base" or config.text_embedding_id == "multilingual-e5-large":
        config.model_id = "Fine-tuned-query"
    else:
        config.model_id = "Fine-tuned"

    print("- " + config.model_id + " - Building the model architecture -")
    train_batches, val_batches, test_batches = data_preparation.get_batches(config)

    fine_tuned_model = text_model.build_my_text_model(config.text_embedding_id, config)

    print("- " + config.model_id + " - Training the model -")
    fine_tuned_model = text_model.train_my_text_model(fine_tuned_model, config.data_id, train_batches, val_batches, config.epochs, config)


    if config.text_model_activate_second_training:
        # Save logs
        fine_tuned_model = text_model.train_my_text_model(fine_tuned_model, config.data_id, train_batches, val_batches, config.epochs, config, unfreeze=True)
        # Save logs

    print("- " + config.model_id + " - Evaluate the model -")
    config.text_eval_start_time = datetime.datetime.now()

    text_evaluation_results = fine_tuned_model.evaluate(test_batches)
    config.text_eval_end_time = datetime.datetime.now()
    log_evaluate_metrics(text_evaluation_results, 1, config)

    global current_trained_model
    current_trained_model = TrainedModel(id=int(config.dir_prefix_id), model_type=0,
                                         value=text_evaluation_results[5],
                                         location=config.output_folder + "/" + config.text_embedding_id)


    return fine_tuned_model

def run_image_branch(config):
    """
    Runs the image branch
    :param config: json config
    """

    config.model_id = "Siamese"

    print("- " + config.model_id + " - Building the model architecture -")
    train_batches, val_batches, test_batches = data_preparation.get_batches(config)

    siamese_model = image_model.build_my_image_model(config)

    print("- " + config.model_id + " - Training the model -")
    siamese_model = image_model.train_my_image_model(siamese_model, config.data_id, train_batches, val_batches, config)

    print("- " + config.model_id + " - Evaluate the model -")
    config.image_eval_start_time = datetime.datetime.now()

    image_evaluation_results = siamese_model.evaluate(test_batches)

    config.image_eval_end_time = datetime.datetime.now()
    log_evaluate_metrics(image_evaluation_results, 2, config)

    global current_trained_model
    current_trained_model = TrainedModel(id=int(config.dir_prefix_id), model_type=1,
                                             value=image_evaluation_results[5],
                                             location=config.output_folder + "/" + config.image_embedding_id)

    return siamese_model


def run_fusion_branch(run_text_model, run_image_model, config):
    """
    Runs the intermediate fusion branch
    :param run_text_model: textmodel
    :param run_image_model: image model
    :param config: json config
    """

    config.model_id = "Intermediate Fusion"
    print("- " + config.model_id + " - Building the model architecture -")
    train_batches, val_batches, test_batches = data_preparation.get_batches(config)

    intermediate_model = fusion_model.build_intermediate_fusion_model(run_text_model, run_image_model, config)

    print("- " + config.model_id + " - Training the model -")
    intermediate_model = fusion_model.train_intermediate_fusion_model(intermediate_model, config.data_id, train_batches, val_batches, config.epochs, config)

    print("- " + config.model_id + " - Evaluate the model -")
    config.fusion_eval_start_time = datetime.datetime.now()

    fusion_evaluation_results = intermediate_model.evaluate(test_batches)
    config.fusion_eval_end_time = datetime.datetime.now()
    log_evaluate_metrics(fusion_evaluation_results, 3, config)

    global current_trained_model
    current_trained_model = TrainedModel(id=int(config.dir_prefix_id), model_type=2, value=fusion_evaluation_results[5],location=config.output_folder + "/" + config.model_id)

    return intermediate_model


def launch(path_to_my_output, config, dir_prefix_id, run_mode):
    """
    Launches the program
    :param path_to_my_output: path to the output folder
    :param config: json config
    :param dir_prefix_id: prefix id
    :param run_mode: selects which branches to execute (1=all, 2=text+image, 3=text, 4=image)
    """
    text_model, image_model, fusion_model = None, None, None

    # Save before redirecting to avoid errors.
    orig_out, orig_err = sys.stdout, sys.stderr

    config.start_date = datetime.datetime.now()
    config.start_date_formated = config.start_date.strftime("%d-%m-%Y_%H:%M:%S")

    # Recognize which runs should be performed

    if run_mode == 1:

        path_to_my_output += "_Fusion"
        model_prefix = "Fusion"

        # prepare
        consol_log_file = create_and_model_dir_and_console_logfile(dir_prefix_id, path_to_my_output, model_prefix ,config)

        path_to_model_config_json = create_run_config_json_copy(path_to_my_output, dir_prefix_id)
        config.output_folder = path_to_my_output
        config.dir_prefix_id = str(dir_prefix_id)

        # start
        text_model = run_text_branch(config)
        image_model = run_image_branch(config)
        fusion_model = run_fusion_branch(text_model, image_model, config)

        # ending
        config.end_date = datetime.datetime.now()
        config.end_date_formated = config.end_date.strftime("%d-%m-%Y_%H:%M:%S")
        write_config_changes_into_json_file(path_to_model_config_json, False, config)
        print("[INFO] End of logfile.")
        sys.stdout, sys.stderr = orig_out, orig_err
        consol_log_file.close()

    elif run_mode == 2:
        # only text and image
        path_to_my_output += "_" + config.text_embedding_id + "_+_" + config.image_embedding_id
        model_prefix = config.text_embedding_id + "_+_" + config.image_embedding_id

        # prepare
        consol_log_file = create_and_model_dir_and_console_logfile(dir_prefix_id, path_to_my_output, model_prefix, config)

        path_to_model_config_json = create_run_config_json_copy(path_to_my_output, dir_prefix_id)
        config.output_folder = path_to_my_output
        config.dir_prefix_id = str(dir_prefix_id)

        # start
        text_model = run_text_branch(config)
        image_model = run_image_branch(config)

        # ending
        config.end_date = datetime.datetime.now()
        config.end_date_formated = config.end_date.strftime("%d-%m-%Y_%H:%M:%S")
        write_config_changes_into_json_file(path_to_model_config_json, False, config)
        print("[INFO] End of logfile.")
        sys.stdout, sys.stderr = orig_out, orig_err
        consol_log_file.close()

    elif run_mode == 3:
        # only text
        path_to_my_output += "_" + config.text_embedding_id
        model_prefix = config.text_embedding_id

        # prepare
        consol_log_file = create_and_model_dir_and_console_logfile(dir_prefix_id, path_to_my_output, model_prefix, config)
        path_to_model_config_json = create_run_config_json_copy(path_to_my_output, dir_prefix_id)
        config.output_folder = path_to_my_output
        config.dir_prefix_id = str(dir_prefix_id)

        # start
        text_model = run_text_branch(config)

        # ending
        config.end_date = datetime.datetime.now()
        config.end_date_formated = config.end_date.strftime("%d-%m-%Y_%H:%M:%S")
        write_config_changes_into_json_file(path_to_model_config_json, False, config)
        print("[INFO] End of logfile.")
        sys.stdout, sys.stderr = orig_out, orig_err
        consol_log_file.close()

    elif run_mode == 4:
        # only image
        path_to_my_output += "_" + config.image_embedding_id
        model_prefix = config.image_embedding_id

        # prepare
        consol_log_file = create_and_model_dir_and_console_logfile(dir_prefix_id, path_to_my_output, model_prefix, config)
        path_to_model_config_json = create_run_config_json_copy(path_to_my_output, dir_prefix_id)
        config.output_folder = path_to_my_output
        config.dir_prefix_id = str(dir_prefix_id)

        # start
        image_model = run_image_branch(config)

        # ending
        config.end_date = datetime.datetime.now()
        config.end_date_formated = config.end_date.strftime("%d-%m-%Y_%H:%M:%S")
        write_config_changes_into_json_file(path_to_model_config_json, False, config)
        print("[INFO] End of logfile.")
        sys.stdout, sys.stderr = orig_out, orig_err
        consol_log_file.close()

    print("[INFO] Finished.")

    return text_model, image_model, fusion_model


def main():

    placeholder_config = Config("programcode/configs/run_config.json")

    # 1 = all (text+image+fusion) (You need to specify a text_embedding_id in the run_config.json)
    # 2 = only text and image (You need to specify a text_embedding_id in the run_config.json)
    # 3 = only text (You need to specify a text_embedding_id in the run_config.json)
    # 4 = only image
    run_mode = 1

    # Number of different configs/runs
    number_of_different_configs = 1

    # Number of runs with the same config
    number_of_runs_within_a_config = 1

    anzahl_aller_runs = number_of_different_configs * number_of_runs_within_a_config
    placeholder_dir_prefix_ids = []
    for z in range(anzahl_aller_runs):
        pending_dir_prefix_id = create_model_dir_placeholder(placeholder_config)
        placeholder_dir_prefix_ids.append(pending_dir_prefix_id)
    placeholder_dir_prefix_ids.sort(reverse=True)

    config = Config("programcode/configs/run_config.json")

    for x in range(number_of_different_configs):
        for y in range(number_of_runs_within_a_config):

            # reset to standards for multiple runs
            config.model_id = None
            config.output_folder = "./myModels/"
            tf.keras.backend.clear_session()

            check_image_for_imagemodel(config)

            # for correct run_id "back-writing"
            dir_prefix_id = placeholder_dir_prefix_ids.pop()
            config.run_id = int(dir_prefix_id)

            path_to_my_output = config.output_folder + dir_prefix_id

            print(f"[INFO] Starting run {(y+1)} with config {x}. ID: {dir_prefix_id}")

            # launch the run
            launch(path_to_my_output, config, dir_prefix_id, run_mode)

            gc.collect()

    print(f"[INFO] All runs finished.")

if __name__ == "__main__":
    main()
