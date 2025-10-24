from . import data_preparation
import tensorflow as tf
import transformers
import datetime
from tensorflow.keras.callbacks import Callback
import tensorflow_addons as tfa

# Metrics for evaluations
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, name="f1_score")
]

# ============================================ build =====================================================================

def build_my_text_model(embedding_id, config):
    """
    Build text model architecture
    :param config:
    :param embedding_id: embedding network to use
    :return: uncompiled model
    """

    # Transformer inputs
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
    attention_masks = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)

    # Load pre-trained embedding network
    if embedding_id.__contains__("xlm-roberta-base"):
        embedding_network = transformers.TFXLMRobertaModel.from_pretrained("xlm-roberta-base")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    elif embedding_id.__contains__("bert-base-multilingual-cased"):
        embedding_network = transformers.TFBertModel.from_pretrained("bert-base-multilingual-cased")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.bert(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    elif embedding_id.__contains__("xlm-roberta-large"):
        embedding_network = transformers.TFXLMRobertaModel.from_pretrained("xlm-roberta-large")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    elif embedding_id.__contains__("paraphrase-xlm-r-multilingual-v1"):
        embedding_network = transformers.TFAutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        embedding_network.trainable = False

        embedding_network_output = embedding_network(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

    elif embedding_id.__contains__("stsb-xlm-r-multilingual"):
        embedding_network = transformers.TFAutoModel.from_pretrained("sentence-transformers/stsb-xlm-r-multilingual")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    elif embedding_id.__contains__("multilingual-e5-small"):
        embedding_network = transformers.TFAutoModel.from_pretrained('intfloat/multilingual-e5-small', from_pt=True)
        embedding_network.trainable = False

        embedding_network_output = embedding_network(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    elif embedding_id.__contains__("multilingual-e5-base"):
        embedding_network = transformers.TFAutoModel.from_pretrained('intfloat/multilingual-e5-base', from_pt=True)
        embedding_network.trainable = False

        embedding_network_output = embedding_network(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    elif embedding_id.__contains__("multilingual-e5-large"):
        embedding_network = transformers.TFAutoModel.from_pretrained('intfloat/multilingual-e5-large', from_pt=True)
        embedding_network.trainable = False

        embedding_network_output = embedding_network(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

    elif embedding_id.__contains__("multilingual-e5-large-instruct"):

        # Loads model as Pytorch instead of Tensorflow model and saves it.
        pt_model = transformers.AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct", use_safetensors=True)
        pt_model.save_pretrained(
            "./myModels/pytorch/e5-pytorch",
            safe_serialization=False
        )

        # Converted to TensorFlow model
        embedding_network = transformers.TFAutoModel.from_pretrained(
            "./myModels/pytorch/e5-pytorch",
            from_pt=True,
            ignore_mismatched_sizes=True
        )
        embedding_network.trainable = False

        embedding_network_output = embedding_network(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

    elif embedding_id.embedding_id.__contains__("RoBERTa"):
        embedding_network = transformers.TFRobertaModel.from_pretrained("roberta-base")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

    elif embedding_id.__contains__("RoBERTa-large"):
        embedding_network = transformers.TFRobertaModel.from_pretrained("roberta-large")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
    else:
        raise RuntimeError("[ERROR] Unknown embedding ID: " + embedding_id)


    # build architecture
    sequence_output = embedding_network_output.last_hidden_state
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.text_architecture_bilstm_units, return_sequences=True))(sequence_output)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    batch_norm = tf.keras.layers.BatchNormalization()(concat)
    dropout_1 = tf.keras.layers.Dropout(config.text_architecture_dropout1)(batch_norm)
    dense_1 = tf.keras.layers.Dense(config.text_architecture_dense1_units, activation=config.text_architecture_dense1_activation, kernel_initializer="he_normal")(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(config.text_architecture_dropout2)(dense_1)
    dense_2 = tf.keras.layers.Dense(config.text_architecture_dense2_units, activation=config.text_architecture_dense2_activation, kernel_initializer="he_normal")(dropout_2)
    dropout_3 = tf.keras.layers.Dropout(config.text_architecture_dropout3)(dense_2)
    output = tf.keras.layers.Dense(1, activation=config.text_architecture_output_activation)(dropout_3)

    model = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=output)

    return model

# ============================================ fit =====================================================================


def fit(model, data_id, train_batches, val_batches, epochs, unfreeze, config):
    """
    Train compiled model
    :param model: model to train
    :param data_id: id of dataset
    :param train_batches: training batches
    :param val_batches: validation batches
    :param epochs: number of epochs
    :param config: json config
    :param unfreeze: should everything be frozen?
    :return: trained model
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.early_stopping_monitor,
        mode=config.early_stopping_mode,
        patience=config.early_stopping_patience,
        restore_best_weights=True
    )

    prio_on_refined_output = 0

    # check, if its the second run with adjusted learning-rate
    if unfreeze:
        # first "run"
        start_epoch_number = config.last_epoch_nr_before_reset + 1
        config.epoch_where_second_lr_started = start_epoch_number
        total_epochs = epochs + config.last_epoch_nr_before_reset + prio_on_refined_output
        config.text_second_training_start_time = datetime.datetime.now()

    else:
        # second "run"
        start_epoch_number = prio_on_refined_output
        total_epochs = epochs + prio_on_refined_output
        config.text_training_start_time = datetime.datetime.now()

    history = model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=total_epochs,
        initial_epoch=start_epoch_number,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[early_stopping],
        verbose=config.training_verbose,    
    )

    if unfreeze:
        config.text_second_training_end_time = datetime.datetime.now()
    else:
        config.text_training_end_time = datetime.datetime.now()


    if early_stopping.stopped_epoch > 0:
        # if ES strikes, save the last epoch (ES -> zero based)
        config.last_epoch_nr_before_reset = early_stopping.stopped_epoch
    else:
        config.last_epoch_nr_before_reset = history.epoch[-1]

    return model

# ============================================ train =====================================================================

def train_my_text_model(model, data_id, train_batches, val_batches, epochs, config, unfreeze=False):
    """
    Train Fine-tuned model
    :param config:
    :param model: model to train
    :param data_id: dataset to use for class weight calculation
    :param train_batches: batches for training
    :param val_batches: batches for validation
    :param epochs: number of epochs
    :param unfreeze: specifies whether to train embedding network
    :return: trained model
    """

    # Set training configuration
    if not unfreeze:
        learning_rate = config.learning_rate
    else:
        model.trainable = True
        learning_rate = config.learning_rate_unfreeze

    if config.text_optimizer == "adamw":
        print("optimizer: AdamW")
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate), loss="BinaryCrossentropy", metrics=metrics)

    elif config.text_optimizer == "sgd":
        print("optimizer: SGD")
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate), loss="BinaryCrossentropy", metrics=metrics)

    model = fit(model, data_id, train_batches, val_batches, epochs, unfreeze, config)

    return model
