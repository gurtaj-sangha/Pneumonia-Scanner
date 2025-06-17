import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Input, Lambda, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def build_resnet50(input_shape=(224, 224, 3), num_classes=3,
                   dropout=0.4, l2_reg=1e-4, lr=1e-4) -> Model:
    ip = Input(shape=input_shape, dtype="float32")
    x  = Lambda(tf.keras.applications.resnet.preprocess_input)(ip)

    base = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation="softmax",
                kernel_regularizer=l2(l2_reg), dtype="float32")(x)

    model = Model(ip, out, name="resnet50_pneumonia")
    model.compile(
        tf.keras.optimizers.Adam(lr),
        tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def unfreeze_top(model: Model, n_conv_layers: int = 50, lr: float = 1e-5):
    conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    for layer in conv[-n_conv_layers:]:
        layer.trainable = True
    model.compile(
        tf.keras.optimizers.Adam(lr),
        tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model