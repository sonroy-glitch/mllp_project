import argparse
import json
import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


def build_datasets(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ]
    )

    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train_ds, val_ds, class_names = build_datasets(args.data_dir)
    model = build_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "sports_model.pkl"), "wb") as handle:
        pickle.dump({"model": model, "class_names": class_names}, handle)

    model.save(os.path.join(args.output_dir, "sports_model.keras"))

    with open(
        os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(class_names, handle, indent=2)

    with open(
        os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, handle, indent=2)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss {loss:.4f} accuracy {accuracy:.4f}")


if __name__ == "__main__":
    main()
