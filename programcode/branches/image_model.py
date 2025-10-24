import tensorflow_addons as tfa
from . import data_preparation
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import Callback

# Path to Tensorflow Swin-Transformer model
# https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224_in22k_fe/1
path_to_swin_model = "SWIN"  # replace with your path to your swin model folder.

# Metrics for evaluations
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, name="f1_score")
]

# ============================================ euclidian distance =====================================================================

def euclidean_distance(vectors):
    """
    Calculate Euclidean distance using Tensorflow
    """
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)

    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


# ============================================ build =====================================================================

def build_my_image_model(config):
    """
    Build model Swin-Transformer with Siamese architecture
    :param config: json config
    :return: uncompiled model
    """

    embedding_network = tf.keras.models.load_model(path_to_swin_model)

    # Unfreeze the last two layers
    for layer in embedding_network.layers:
        if layer.name != "basic_layer_3":
            layer.trainable = False

    for layer in embedding_network.get_layer("basic_layer_3").layers:
        if layer.name != "swin_transformer_block_1":
            layer.trainable = False

    for layer in embedding_network.get_layer("basic_layer_3").get_layer("swin_transformer_block_1").layers:
        if layer.name != "mlp":
            layer.trainable = False

    for layer in embedding_network.get_layer("basic_layer_3").get_layer("swin_transformer_block_1").get_layer("mlp").layers:
        if layer.name not in ["dense_49", "dense_50"]:
                layer.trainable = False
        else:
                layer.trainable = True

    # Build sub-network architecture
    image = tf.keras.layers.Input((config.image_height, config.image_width) + (3,))
    embedding = embedding_network(image, training=False)

    dropout = tf.keras.layers.Dropout(config.image_architecture_dropout)(embedding)
    dense_1 = tf.keras.layers.Dense(config.image_architecture_dense1_units,
                                    activation=config.image_architecture_dense1_activation,
                                    kernel_initializer="he_normal")(dropout)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(dense_1)
    dense_2 = tf.keras.layers.Dense(config.image_architecture_dense2_units,
                                    activation=config.image_architecture_dense2_activation,
                                    kernel_initializer="he_normal")(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(dense_2)
    sister_output = tf.keras.layers.Dense(config.image_architecture_sister_output_dense)(batch_norm_2)
    sister_model = tf.keras.Model(image, sister_output)

    # Build siamese architecture
    image_1 = tf.keras.layers.Input((config.image_height, config.image_width) + (3,))
    sister_1 = sister_model(image_1)

    image_2 = tf.keras.layers.Input((config.image_height, config.image_width) + (3,))
    sister_2 = sister_model(image_2)

    merge = tf.keras.layers.Lambda(euclidean_distance)([sister_1, sister_2])
    batch_norm_3 = tf.keras.layers.BatchNormalization()(merge)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(batch_norm_3)
    model = tf.keras.Model(inputs=[image_1, image_2], outputs=output)

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

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.early_stopping_monitor, 
        mode=config.early_stopping_mode, 
        patience=config.early_stopping_patience,
        restore_best_weights=True
        )  

    config.image_training_start_time = datetime.datetime.now()

    history = model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=epochs,
        initial_epoch=0,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[early_stopping],
        verbose=config.training_verbose,
    )

    config.image_training_end_time = datetime.datetime.now()

    if early_stopping.stopped_epoch > 0:
        # if ES strikes, save the last epoch (ES -> zero based)
        config.last_epoch_nr_before_reset = early_stopping.stopped_epoch
    else:
        config.last_epoch_nr_before_reset = history.epoch[-1] if history.epoch else epochs - 1

    return model

# ============================================ loss ============================================================

def loss(margin=1):
    """
    Calculate contrastive loss
    """
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

# ============================================ train ============================================================

def train_my_image_model(model, data_id, train_batches, val_batches, config):
    """
    Train Siamese model
    :param model: model to train
    :param data_id: id of the dataset
    :param train_batches: training batches
    :param val_batches: validation batches
    :param config: json config
    :return: trained model
    """

    if config.image_learning_rate == "" or config.image_learning_rate == 0:
        raise RuntimeError("[ERROR] The provided learning rate is empty. The program will exit.")

    if config.image_optimizer == "adamw":
        # use adam as optimizer
        print("[INFO] Training Siamese model with optimizer Adam_W")
        optimizer = tf.keras.optimizers.AdamW(config.image_learning_rate)
    else:
        # default optimizer RMSprop
        print("[INFO] Training Siamese model with optimizer RMSprop")
        optimizer = tf.keras.optimizers.RMSprop(config.image_learning_rate)

    model.compile(loss=loss(margin=1), optimizer=optimizer, metrics=metrics)
    model = fit(model, data_id, train_batches, val_batches, config.epochs, config)

    return model
