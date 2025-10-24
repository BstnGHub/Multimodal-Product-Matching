from . import data_preparation
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
from tensorflow.keras.callbacks import Callback

# Metrics for evaluations
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, name="f1_score")
]

# ============================================ build =====================================================================

def build_intermediate_fusion_model(fine_tuned_model, siamese_model, config):
    """
    Build model with Intermediate Fusion architecture
    :param fine_tuned_model: pre-trained Text Branch
    :param siamese_model: pre-trained Image Branch
    :param config: json config
    :return: uncompiled model
    """

    fine_tuned_model.trainable = False

    avg_pool = fine_tuned_model.get_layer("concatenate").input[0]
    max_pool = fine_tuned_model.get_layer("concatenate").input[1]

    siamese_model.trainable = False

    image_embedding_1 = siamese_model.get_layer("lambda").input[0]
    image_embedding_2 = siamese_model.get_layer("lambda").input[1]

    # Build architecture
    concat = tf.keras.layers.concatenate([image_embedding_1, image_embedding_2, avg_pool, max_pool])
    batch_norm = tf.keras.layers.BatchNormalization()(concat)

    dropout_1 = tf.keras.layers.Dropout(config.fusion_architecture_dropout1)(batch_norm)
    dense_1 = tf.keras.layers.Dense(config.fusion_architecture_dense1_units, activation=config.fusion_architecture_dense1_activation, kernel_initializer="he_normal")(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(config.fusion_architecture_dropout2)(dense_1)
    dense_2 = tf.keras.layers.Dense(config.fusion_architecture_dense2_units, activation=config.fusion_architecture_dense2_activation, kernel_initializer="he_normal")(dropout_2)
    dropout_3 = tf.keras.layers.Dropout(config.fusion_architecture_dropout3)(dense_2)
    dense_3 = tf.keras.layers.Dense(config.fusion_architecture_dense3_units, activation=config.fusion_architecture_dense3_activation, kernel_initializer="he_normal")(dropout_3)
    dropout_4 = tf.keras.layers.Dropout(config.fusion_architecture_dropout4)(dense_3)
    output = tf.keras.layers.Dense(1, activation=config.fusion_architecture_output_activation)(dropout_4)

    # Rename layers to have unique names
    for layer in siamese_model.layers[:2]:
        layer._name = layer.name + str("_2")

    model = tf.keras.models.Model(inputs=fine_tuned_model.input + siamese_model.input, outputs=output)

    return model

# ============================================ fit =====================================================================

def fit(model, data_id, train_batches, val_batches, epochs, config):
    """
    Train compiled model
    :param model: model to train
    :param data_id: id of dataset
    :param train_batches: training batches
    :param val_batches: validation batches
    :param epochs: number of epochs
    :param config: json config
    :return: trained model
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=config.early_stopping_monitor, mode=config.early_stopping_mode, patience=config.early_stopping_patience, restore_best_weights=True)

    config.fusion_training_start_time = datetime.datetime.now()

    model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[early_stopping],
        verbose=config.training_verbose,
    )
    config.fusion_training_end_time = datetime.datetime.now()

    return model

# ============================================ train =====================================================================

def train_intermediate_fusion_model(model, data_id, train_batches, val_batches, epochs, config):
    """
    Train Intermediate Fusion model (or Valenciano et al.'s model)
    :param model: model to train
    :param data_id: dataset to use for class weight calculation
    :param train_batches: batches for training
    :param val_batches: batches for validation
    :param epochs: number of epochs
    :param config: configuration
    :return: trained model
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="BinaryCrossentropy", metrics=metrics)
    model = fit(model, data_id, train_batches, val_batches, epochs, config)

    return model
