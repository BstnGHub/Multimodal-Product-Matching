# Prompt or Train? — A Comparative Benchmark of VLM Prompting and Intermediate Fusion for Product Matching

This is the official GitHub repository for the paper **“Prompt or Train? A Comparative Benchmark of VLM Prompting and Intermediate Fusion for Product Matching.”**  

# How-to-Use

## Contents

### WDC Shoes dataset

* `wdc_shoes_data.pickle`: Training, validation, and test sets (each contains **pairs of texts**, **pairs of image paths**, and **labels**).
* `wdc_show_images/`: Folder with **WDC Shoes** images.

### Zalando dataset

* `zalando_data.pickle`: Training, validation, and test sets (each contains **pairs of texts**, **pairs of image paths**, and **labels**).
* `zalando_images/`: Folder with **Zalando** images.

### ProMap dataset

* `promap_data.pickle`: Training, validation, and test sets (each contains **pairs of texts**, **pairs of image paths**, and **labels**).
* `promap_images/`: Folder with **ProMap** images.


## Setup

- install the requirements with (`pip install -r requirements.txt`)
- Download the Tensorflow Swin-Transformer model: https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224_in22k_fe/1


The program is controlled on two levels: a set of central parameters in **`main.py`** and the detailed configuration in the **`run_config.json`**.

---

## Parameters in `main.py`

- **`number_of_different_configs`**: Specifies how many different configuration files (`run_config.json`) are executed sequentially. Default is `1`.
- **`number_of_runs_within_a_config`**: Determines how many complete runs are performed with the same configuration.
  - Example: If set to `3`, the given run_config.json will be trained and evaluated three times.
- **`run_mode`**: Controls which model branches are executed in a run:

  - **`1`**: Text model, image model, and fusion are trained and evaluated.  
    Requires both `text_embedding_id` and `image_embedding_id`. The fusion phase will run after text and image training.

  - **`2`**: Only text and image models are executed, **without** fusion.  
    Requires both `text_embedding_id` and `image_embedding_id`. 

  - **`3`**: Only the text model is executed.  
    Requires `text_embedding_id`. Image and fusion are skipped.

  - **`4`**: Only the image model is executed.  
    Requires `image_embedding_id`. Text and fusion are skipped.

---

## run_config.json Parameters

### General Information
- **`run_id`**: Unique run identifier. Automatically incremented to ensure no two runs share the same ID (e.g., Run 1000 → Run 1001). Always takes precedence over run IDs stored in model folders.
- **`start_date` / `start_date_formated`**: Exact and formatted timestamp marking the run start.
- **`end_date` / `end_date_formated`**: Exact and formatted timestamp marking the run end.

### Training and Evaluation Durations
- **`text_training_time` / `text_second_training_time` / `image_training_time` / `fusion_training_time`**: Duration of training phases (in minutes), automatically recorded after the run.
- **`text_eval_time` / `image_eval_time` / `fusion_eval_time`**: Duration of evaluation phases (in minutes), automatically recorded after the run.

### Training Configuration
- **`epochs`**: Maximum number of training epochs.
- **`batch_size`**: Number of samples per training batch.

### Model Selection
- **`text_embedding_id` / `image_embedding_id`**: Specify the text and/or image model to be used.
  - Only text model set → trains text model only.
  - Only image model set → trains image model only.
  - Both set → first trains text model, then image model.
  
- Currently supported **`text_embedding_id`** models are:
  - RoBERTa-large
  - bert-base-multilingual-cased
  - xlm-roberta-large
  - multilingual-e5-small /-base /-large /-large-instruct

- Currently, the only supported **`image_embedding_id`** model is the Swin-Transformer.

### Internal Flags and Metrics
- **Metrics (`text_loss`, `text_accuracy`, `text_precision`, `text_recall`, `text_auc`, `text_f1`, `image_loss`, `image_accuracy`, `image_precision`, `image_recall`, `image_auc`, `image_f1`, `fusion_loss`, `fusion_accuracy`, `fusion_precision`, `fusion_recall`, `fusion_auc`, `fusion_f1`)**: Automatically set by the program. Default value is `null`. They serve purely informational purposes after a run and must not be modified manually.

### System and Hardware
- **`gpu`**: Optional field to document the GPU used for the run.
- **`system`**: Labels the execution environment (e.g., *cluster*, *cluster capella*). If set to **`local`**, all console logs are written directly to the console (recommended for debugging). Default: *cluster*.
- **`training_verbose`**: Controls verbosity of training output (as in TensorFlow/Keras).
  - `0`: no output
  - `1`: progress bar per epoch (default)
  - `2`: one line per epoch

### Dataset and Run Control
- **`data_id`**: Defines which dataset to use. Supported values: **`Zalando`**, **`WDC Shoes`**.
- **`output_folder`**: Directory for runtime artefacts (configuration, logs, results).
- **`dir_prefix_id`**: Internally set by the program; must not be modified.

### Early Stopping and Learning Rates
- **`early_stopping_monitor`**: Metric monitored for early stopping (e.g., `val_f1_score`).
- **`early_stopping_mode`**: Direction for optimization (`max` or `min`).
- **`early_stopping_patience`**: Number of epochs without improvement before training stops.
- **`learning_rate`**: Initial learning rate.
- **`learning_rate_unfreeze`**: Learning rate applied after unfreezing.
- **`epoch_where_second_lr_started`**: Logging parameter only. Default `-1`, must not be changed.
- **`image_learning_rate`**: Learning rate for the image model.
- **`image_optimizer` / `text_optimizer`**: Optimizer for image and text models (e.g., `adamw`).
- **`text_model_activate_second_training`**: Enables a second training phase for the text model with the unfreeze learning rate. Default: `true`.

### Architecture Configuration
- **Architecture parameters (`text_architecture_*`, `image_architecture_*`, `fusion_architecture_*`)**: Allow modifications to model architectures, such as layer sizes, activation functions, or dropout rates, for text, image, and fusion models.

### Timing Parameters
- **Timing parameters (`text_training_start_time` … `fusion_eval_end_time`, plus `text_second_training_*`)**: Automatically set by the program. Indicate the duration (in minutes) of individual training and evaluation phases. Must not be modified by the user.
