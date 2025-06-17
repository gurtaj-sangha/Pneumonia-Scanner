from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)

from data_pipeline import make_generators, class_weights
from model_builder import build_resnet50, unfreeze_top

OUT_DIR = Path("models") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_gen, val_gen, test_gen = make_generators(batch=32)
weights = class_weights(train_gen)

model = build_resnet50(lr=1e-4)

cbs = [
    ModelCheckpoint(OUT_DIR / "best.h5", monitor="val_auc",
                    mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max",
                  patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_auc", mode="max",
                      patience=2, factor=0.2, verbose=1),
]

# phase-1: head only
model.fit(train_gen, epochs=5, validation_data=val_gen,
          class_weight=weights, callbacks=cbs)

# phase-2: fine-tune top conv layers
model = unfreeze_top(model, n_conv_layers=50, lr=3e-5)
#EarlyStopping(patience=6, monitor="val_auc", mode="max")
#ReduceLROnPlateau(patience=3, factor=0.3)
model.fit(train_gen, epochs=5, validation_data=val_gen,
          class_weight=weights, callbacks=cbs)

print("\nTest metrics:")
for n, v in zip(model.metrics_names, model.evaluate(test_gen, verbose=0)):
    print(f"{n}: {v:.4f}")

model.save(OUT_DIR / "final.h5")
print(f"Saved -> {OUT_DIR / 'final.h5'}")
