import pickle
import tensorflow as tf
import transformers
import numpy as np
from PIL import Image
from configs.config import Config

def load_wdc_shoes_data():
    """
    Load WDC Shoes dataset from pickle file
    """

    train_data, val_data, test_data = pickle.load(open("datasets/wdc_shoes_data.pickle", "rb"))

    return train_data, val_data, test_data

def load_zalando_data():
    """
    Load Zalando dataset from pickle file
    """
    train_data, val_data, test_data = pickle.load(open("datasets/zalando_data.pickle", "rb"))

    return train_data, val_data, test_data

def load_promap_data():
    """
    Load ProMap dataset from pickle file
    """
    train_data, val_data, test_data = pickle.load(open("datasets/promap_data.pickle", "rb"))

    return train_data, val_data, test_data


def get_class_weights(data_id):
    """
    Get class weights for a given data id
    """

    # Load dataset
    if data_id == "WDC Shoes":
        train_data, val_data, test_data = load_wdc_shoes_data()
    elif data_id == "Zalando":
        train_data, val_data, test_data = load_zalando_data()
    elif data_id == "ProMap":
        train_data, val_data, test_data = load_promap_data()
    else:
        raise RuntimeError("[ERROR] Unknown data_id: " + str(data_id))

    # Calculate weights
    neg = train_data["labels"].count(0)
    pos = train_data["labels"].count(1)
    total = neg + pos

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    return {0: weight_for_0, 1: weight_for_1}


# Load and preprocess images
def preprocess_image(filename, height, width):
    """"
    Load and preprocess images
    """
    with Image.open(filename) as img:
        # Force three channels so the batcher always yields consistent shape.
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
            image = np.array(img)[..., :3]
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = np.array(img)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, height, width)  # default 224, 224
    return image

def encode_texts(tokenizer, text_embedding_id, text_pairs):
    """
    Encode product-descriptions.
    """
    if (
            text_embedding_id != "multilingual-e5-large-instruct"
            and text_embedding_id != "multilingual-e5-small"
            and text_embedding_id != "multilingual-e5-base"
            and text_embedding_id != "multilingual-e5-large"
    ):
        text_pairs = text_pairs.tolist()
    
    if isinstance(text_pairs, np.ndarray):
        text_pairs = text_pairs.tolist()

    encoded = tokenizer.batch_encode_plus(
        text_pairs,
        add_special_tokens=True,
        max_length=128,
        return_attention_mask=True,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )

    input_ids = np.array(encoded["input_ids"], dtype="int32")
    attention_masks = np.array(encoded["attention_mask"], dtype="int32")
    token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

    return input_ids, attention_masks, token_type_ids


def encode_query_text_instruct(text_pairs, prefix=None):
    """
    Encode texts for multilingual-e5-large-instruct, because it allows instructions.
    For more information, visit: https://huggingface.co/intfloat/multilingual-e5-large-instruct
    """

    pairs = text_pairs.tolist()

    # Instruction and prefixes
    task = "How similar are the two product descriptions"
    query_tag = "Query:"
    passage_tag = "passage:"  # or "document:"

    # Formatting using list compression
    formatted_pairs = [
        [
            f"Instruct: {task}\n{query_tag} {q}",
            f"{passage_tag} {d}"
        ]
        for q, d in pairs
    ]

    return formatted_pairs

def encode_query_text(text_pairs):
    """
    Encode texts for multilingual-e5-small/base/large, because they allow a query.
    For more information, visit: https://huggingface.co/intfloat/multilingual-e5-large
    """

    return [
        [f"query: {q}", f"passage: {d}"]  # Query vs. Passage
        for q, d in text_pairs
    ]

class BatchGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            model_id,
            text_embedding_id,
            image_embedding_id,
            text_pairs,
            image_pairs,
            labels,
            image_height,
            image_width,
            batch_size,
            shuffle=True,
            include_targets=True,
            data_id=None,
    ):
        """
        Generate batches for training, validation or testing
        :param model_id: model to generate inputs for
        :param text_embedding_id: text embedding network to tokenize texts for
        :param text_pairs: text pairs to encode
        :param image_pairs: image pairs to load and preprocess
        :param labels: labels to use for validation and testing
        :param shuffle: specifies whether to shuffle pairs and labels
        :param include_targets: specifies whether to include labels
        """
        self.model_id = model_id
        self.text_pairs = text_pairs
        self.image_pairs = image_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.text_embedding_id = text_embedding_id
        self.image_embedding_id = image_embedding_id
        self.include_targets = include_targets
        self.data_id = data_id

        if text_embedding_id.__contains__("RoBERTa-large"):
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-large")
        elif text_embedding_id.__contains__("xlm-roberta-base"):
            self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        elif text_embedding_id.__contains__("xlm-roberta-large"):
            self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        elif text_embedding_id.__contains__("bert-base-multilingual-cased"):
            self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        elif text_embedding_id.__contains__("multilingual-e5-small"):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        elif text_embedding_id.__contains__("multilingual-e5-base"):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
        elif text_embedding_id.__contains__("multilingual-e5-large"):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        elif text_embedding_id.__contains__("multilingual-e5-large-instruct"):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        elif text_embedding_id.__contains__("RoBERTa"):
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")

        self.indexes = np.arange(len(self.text_pairs))

        self.on_epoch_end()

    def __len__(self):
        return len(self.text_pairs) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        text_pairs = self.text_pairs[indexes]
        image_pairs = self.image_pairs[indexes]

        # Prepare inputs
        inputs = []
        if self.model_id == "Fine-tuned" or self.model_id == "Intermediate Fusion":
            input_ids, attention_masks, token_type_ids = encode_texts(self.tokenizer, self.text_embedding_id,
                                                                      text_pairs)

            inputs = [input_ids, attention_masks, token_type_ids]
        elif self.model_id == "Fine-tuned-query-instruct":

            text_pairs = encode_query_text_instruct(text_pairs, "query:")
            input_ids, attention_masks, token_type_ids = encode_texts(self.tokenizer, self.text_embedding_id,
                                                                      text_pairs)

            inputs = [input_ids, attention_masks, token_type_ids]
        elif self.model_id == "Fine-tuned-query":

            text_pairs = encode_query_text(text_pairs)
            input_ids, attention_masks, token_type_ids = encode_texts(self.tokenizer, self.text_embedding_id,
                                                                      text_pairs)

            inputs = [input_ids, attention_masks, token_type_ids]

        if self.model_id == "Siamese" or self.model_id == "Intermediate Fusion":

            images_1 = [preprocess_image(image_pair[0], self.image_height, self.image_width) for image_pair in
                            image_pairs]
            images_2 = [preprocess_image(image_pair[1], self.image_height, self.image_width) for image_pair in
                            image_pairs]

            inputs += [np.array(images_1, dtype="float32"), np.array(images_2, dtype="float32")]

        # Return batches
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="float32")

            return inputs, labels
        else:
            return inputs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
        

def get_batches(config):
    """
    Generate batches for training, validation and testing
    """
    if config.data_id == "WDC Shoes":
        train_data, val_data, test_data = load_wdc_shoes_data()
    elif config.data_id == "Zalando":
        train_data, val_data, test_data = load_zalando_data()
    elif config.data_id == "ProMap":
        train_data, val_data, test_data = load_promap_data()
    else:
        raise RuntimeError("[ERROR] Unknown data_id: " + str(config.data_id))

    train_batches = BatchGenerator(
        config.model_id,
        config.text_embedding_id,
        config.image_embedding_id,
        np.array(train_data["text_pairs"]),
        np.array(train_data["image_pairs"]),
        np.array(train_data["labels"]),
        config.image_height,
        config.image_width,
        config.batch_size,
        shuffle=True,
        data_id=config.data_id,
    )
    val_batches = BatchGenerator(
        config.model_id,
        config.text_embedding_id,
        config.image_embedding_id,
        np.array(val_data["text_pairs"]),
        np.array(val_data["image_pairs"]),
        np.array(val_data["labels"]),
        config.image_height,
        config.image_width,
        config.batch_size,
        shuffle=False,
        data_id=config.data_id,
    )
    test_batches = BatchGenerator(
        config.model_id,
        config.text_embedding_id,
        config.image_embedding_id,
        np.array(test_data["text_pairs"]),
        np.array(test_data["image_pairs"]),
        np.array(test_data["labels"]),
        config.image_height,
        config.image_width,
        config.batch_size,
        shuffle=False,
        data_id=config.data_id,
    )

    return train_batches, val_batches, test_batches