"""Backbone, sigmoid head, weighted loss, and the stage-1 / stage-2 freeze rules."""

from __future__ import annotations

import keras
import tensorflow as tf

try:
    from keras.saving import register_keras_serializable
except ImportError:  # older keras
    from keras.utils import register_keras_serializable

from .labels import N_LABELS

BACKBONE_BUILDERS = {
    "EfficientNetB0": "efficientnet.EfficientNetB0",
    "DenseNet201": "densenet.DenseNet201",
    "ResNet50": "resnet.ResNet50",
    "InceptionV3": "inception_v3.InceptionV3",
    "Xception": "xception.Xception",
}

# Global batch size before it is rounded to the GPU count. DenseNet201 and
# Xception are the heavy pair on a 16 GB T4.
DEFAULT_BATCH = {
    "EfficientNetB0": 32,
    "ResNet50": 32,
    "InceptionV3": 32,
    "Xception": 16,
    "DenseNet201": 16,
}

HEAD_LAYERS = {"gap", "head_dropout", "probs"}
WEIGHT_CAP = 30.0


def builder_for(backbone: str):
    if backbone not in BACKBONE_BUILDERS:
        known = ", ".join(sorted(BACKBONE_BUILDERS))
        raise ValueError(f"Unknown BACKBONE {backbone!r}. Choose one of: {known}")
    module_name, class_name = BACKBONE_BUILDERS[backbone].split(".")
    module = __import__(f"keras.applications.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)


@register_keras_serializable(package="cxr")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Inverse-frequency binary cross-entropy, one weight pair per label.

    For label j, with N training images and N_pos positives:
        w_pos = N / (2 * N_pos), w_neg = N / (2 * N_neg)
    Weights are clipped so a rare label such as Hernia cannot dominate.
    """

    def __init__(self, pos_weight, neg_weight, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = [float(value) for value in pos_weight]
        self.neg_weight = [float(value) for value in neg_weight]

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        pos = tf.constant(self.pos_weight, dtype=tf.float32)
        neg = tf.constant(self.neg_weight, dtype=tf.float32)
        loss = -(pos * y_true * tf.math.log(y_pred) + neg * (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config["pos_weight"] = self.pos_weight
        config["neg_weight"] = self.neg_weight
        return config


def class_weights(train_labels) -> tuple[list[float], list[float]]:
    """train_labels is a dataframe or array of shape (N, n_labels) with 0/1 values."""
    import numpy as np

    matrix = np.asarray(train_labels, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_LABELS:
        raise ValueError(f"Expected train labels of shape (N, {N_LABELS}), got {matrix.shape}")
    n = float(matrix.shape[0])
    n_pos = matrix.sum(axis=0)
    n_neg = n - n_pos
    pos = []
    neg = []
    for index, label_pos in enumerate(n_pos):
        if label_pos < 1 or n_neg[index] < 1:
            print(f"Warning: label {index} has pos={label_pos:.0f} neg={n_neg[index]:.0f}. Using weight 1.")
            pos.append(1.0)
            neg.append(1.0)
            continue
        w_pos = n / (2.0 * label_pos)
        w_neg = n / (2.0 * n_neg[index])
        if w_pos > WEIGHT_CAP or w_neg > WEIGHT_CAP:
            print(f"Clipping label {index} weights from pos={w_pos:.1f} neg={w_neg:.1f} to {WEIGHT_CAP:.0f}.")
        pos.append(float(min(w_pos, WEIGHT_CAP)))
        neg.append(float(min(w_neg, WEIGHT_CAP)))
    return pos, neg


def build_model(backbone: str, weights: str | None):
    """ImageNet weights or None. The head is a new sigmoid over the fixed label order.

    input_tensor keeps every conv layer on one graph so Grad-CAM can see them
    and so freezing is applied to real layers rather than a nested model.
    """
    inputs = keras.Input(shape=(224, 224, 3), name="cxr_image")
    base = builder_for(backbone)(include_top=False, weights=weights, input_tensor=inputs)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = keras.layers.Dropout(0.2, name="head_dropout")(x)
    # float32 so mixed precision does not pass a float16 probability to the loss.
    outputs = keras.layers.Dense(N_LABELS, activation="sigmoid", dtype="float32", name="probs")(x)
    model = keras.Model(inputs, outputs, name=backbone)
    if "float32" not in str(model.output.dtype):
        raise AssertionError(f"Head dtype is {model.output.dtype}, expected float32")
    return model


def walk_layers(layer, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if id(layer) in seen:
        return
    seen.add(id(layer))
    yield layer
    children = getattr(layer, "layers", None)
    if children:
        for child in children:
            yield from walk_layers(child, seen)


def is_batchnorm(layer) -> bool:
    if isinstance(layer, keras.layers.BatchNormalization):
        return True
    return layer.__class__.__name__ in {"BatchNormalization", "SyncBatchNormalization"}


def is_conv(layer) -> bool:
    return isinstance(
        layer,
        (keras.layers.Conv2D, keras.layers.SeparableConv2D, keras.layers.DepthwiseConv2D),
    )


def ordered_layers(model: keras.Model) -> list:
    top = list(model.layers)
    if sum(1 for layer in top if is_conv(layer)) >= 5:
        return top
    return list(walk_layers(model))


def freeze_batchnorm(model: keras.Model) -> int:
    frozen = 0
    for layer in walk_layers(model):
        if is_batchnorm(layer):
            layer.trainable = False
            frozen += 1
    if frozen == 0:
        raise AssertionError("No BatchNormalization layers found to freeze.")
    return frozen


def freeze_backbone(model: keras.Model) -> None:
    for layer in ordered_layers(model):
        if layer.name in HEAD_LAYERS:
            continue
        layer.trainable = False
    model.get_layer("probs").trainable = True
    freeze_batchnorm(model)


def unfreeze_top(model: keras.Model, fraction: float = 0.30) -> int:
    """Unfreeze the top fraction of backbone layers. BatchNorm stays frozen."""
    body = []
    for layer in ordered_layers(model):
        if layer.name in HEAD_LAYERS:
            continue
        if isinstance(layer, keras.layers.InputLayer):
            continue
        body.append(layer)
    if not body:
        raise AssertionError("Backbone has no layers to unfreeze.")
    n_top = max(1, int(round(len(body) * fraction)))
    for layer in body[:-n_top]:
        layer.trainable = False
    for layer in body[-n_top:]:
        layer.trainable = True
    model.get_layer("probs").trainable = True
    freeze_batchnorm(model)
    return n_top


def assert_stage1(model: keras.Model) -> None:
    convs = [layer for layer in walk_layers(model) if is_conv(layer)]
    if not convs:
        raise AssertionError("Model has no conv layers.")
    if any(layer.trainable for layer in convs):
        trainable = [layer.name for layer in convs if layer.trainable][:8]
        raise AssertionError(f"Stage 1 left conv layers trainable: {trainable}")
    if not model.get_layer("probs").trainable:
        raise AssertionError("Stage 1 froze the new head.")
    if any(layer.trainable for layer in walk_layers(model) if is_batchnorm(layer)):
        raise AssertionError("Stage 1 left a BatchNorm layer trainable.")


def assert_stage2(model: keras.Model) -> None:
    convs = [layer for layer in walk_layers(model) if is_conv(layer)]
    if not any(layer.trainable for layer in convs):
        raise AssertionError("Stage 2 did not unfreeze any conv layer.")
    if not any(not layer.trainable for layer in convs):
        raise AssertionError("Stage 2 unfroze the entire backbone; the bottom should stay frozen.")
    if any(layer.trainable for layer in walk_layers(model) if is_batchnorm(layer)):
        raise AssertionError("Stage 2 left a BatchNorm layer trainable.")
    if not model.get_layer("probs").trainable:
        raise AssertionError("Stage 2 froze the head.")


def trainable_parameter_count(model: keras.Model) -> int:
    return int(sum(tf.size(weight).numpy() for weight in model.trainable_weights))


def find_last_conv(model: keras.Model):
    last = None
    for layer in walk_layers(model):
        if is_conv(layer):
            last = layer
    if last is None:
        raise AssertionError("No conv layer found for Grad-CAM.")
    return last


def compile_model(model: keras.Model, optimizer, loss) -> None:
    """Compile after the freeze flags are set, then freeze BatchNorm again.

    Keras can flip BatchNorm back to trainable when the parent changes. A
    second compile picks up the corrected flags before any gradient step.
    """
    freeze_batchnorm(model)
    model.compile(optimizer=optimizer, loss=loss)
    if any(is_batchnorm(layer) and layer.trainable for layer in walk_layers(model)):
        freeze_batchnorm(model)
        model.compile(optimizer=optimizer, loss=loss)


CUSTOM_OBJECTS = {"WeightedBinaryCrossentropy": WeightedBinaryCrossentropy}


def align_batch_size(batch_size: int, n_gpus: int) -> int:
    replicas = max(int(n_gpus), 1)
    batch_size = max(int(batch_size), replicas)
    return batch_size - (batch_size % replicas)
