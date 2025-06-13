"""
train_resnet50.py
=================
End-to-end training script for three-way pneumonia classification
using ResNet-50 with a two-phase fine-tuning schedule.

Prereqs
-------
pip install tensorflow matplotlib scikit-learn
(make sure your PneumoniaDataPipeline class lives in src/data_pipeline.py)

Usage
-----
python train_resnet50.py
"""

from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Input, Lambda, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, CSVLogger)

# ----------------------------------------------------------------------
# 0.  REPRODUCIBILITY
# ----------------------------------------------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# ----------------------------------------------------------------------
# 1.  DATA PIPELINE  (import the class you wrote earlier)
# ----------------------------------------------------------------------
from data_pipeline import PneumoniaDataPipeline  # adjust path as needed

pipeline = PneumoniaDataPipeline(batch_size=16)   # uses config.DATA_DIR
train_gen, val_gen, _ = pipeline.create_data_generators()
class_weights = pipeline.calculate_class_weights()

# ----------------------------------------------------------------------
# 2.  BUILD  +  COMPILE  (ResNet-50, ImageNet weights, preprocessing)
# ----------------------------------------------------------------------
def build_model(input_shape=(224, 224, 3), num_classes=3,
                l2_reg=1e-4, dropout_rate=0.4) -> Model:

    preprocess_input = tf.keras.applications.resnet.preprocess_input
    inputs = Input(shape=input_shape, dtype="float32", name="input")
    x = Lambda(preprocess_input, name="preprocess")(inputs)

    base = ResNet50(include_top=False, weights="imagenet",
                    input_tensor=x)
    base.trainable = False                      # phase-1: freeze backbone

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(dropout_rate, name="dropout")(x)
    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_regularizer=l2(l2_reg),
                    dtype="float32",
                    name="predictions")(x)

    model = Model(inputs, outputs, name="resnet50_pneumonia")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                 tf.keras.metrics.Recall(name="recall", thresholds=0.5)],
    )
    return model


model = build_model()

# ----------------------------------------------------------------------
# 3.  CALLBACKS
# ----------------------------------------------------------------------
Path("models").mkdir(exist_ok=True)
tag = datetime.now().strftime("%Y%m%d_%H%M")

cbs = [
    ModelCheckpoint(f"models/best_resnet50_{tag}.h5",
                    monitor="val_auc", mode="max",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max",
                  patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_auc", mode="max",
                      factor=0.2, patience=2, verbose=1),
    CSVLogger(f"training_log_{tag}.csv")
]

# ----------------------------------------------------------------------
# 4.  PHASE-1  TRAIN  (frozen backbone)
# ----------------------------------------------------------------------
EPOCHS_WARMUP = 5
history_warm = model.fit(
    train_gen,
    epochs=EPOCHS_WARMUP,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=cbs,
)

# ----------------------------------------------------------------------
# 5.  UNFREEZE LAST 50 CONV LAYERS  +  RE-COMPILE
# ----------------------------------------------------------------------
for layer in model.layers[-50:]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision", thresholds=0.5),
             tf.keras.metrics.Recall(name="recall", thresholds=0.5)],
)

# ----------------------------------------------------------------------
# 6.  PHASE-2  FINE-TUNE
# ----------------------------------------------------------------------
EPOCHS_FINETUNE = 10
history_fine = model.fit(
    train_gen,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=cbs,
)

# ----------------------------------------------------------------------
# 7.  SAVE FINAL MODEL
# ----------------------------------------------------------------------
final_path = f"models/resnet50_final_{tag}.h5"
model.save(final_path)
print(f"\nTraining complete → saved final model to {final_path}")

