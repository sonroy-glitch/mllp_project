# Model folder

Place the trained classifier here. The loader looks for these names, in order:

1. `sports_model.pkl`
2. `model.pkl`
3. `sports_model.keras` / `model.keras`
4. `sports_model.h5` / `model.h5`

If none match, the first `.pkl`, `.keras` or `.h5` file in this folder is used.

## Pickle format

Two layouts are accepted.

**Preferred — a dictionary carrying the labels with the weights:**

```python
pickle.dump({"model": model, "class_names": class_names}, open("sports_model.pkl", "wb"))
```

**Bare model object:**

```python
pickle.dump(model, open("sports_model.pkl", "wb"))
```

With the bare form the labels must come from `class_names.json` in this folder,
otherwise the defaults in `backend/sports_metadata.py` are used.

## class_names.json

A JSON array in the exact index order the model was trained on:

```json
["Badminton", "Cricket", "Karate", "Soccer", "Swimming", "Tennis", "Wrestling"]
```

`tf.keras.utils.image_dataset_from_directory` sorts class folders alphabetically, so
`train_ds.class_names` already holds the correct order. The seven classes above are
alphabetical, which is why the list looks different from the frequency order the
dataset is usually quoted in (Cricket 1556, Wrestling 1471, Tennis 1445, Badminton
1394, Soccer 1188, Swimming 595, Karate 578).

## Expected contract

| Property | Value |
| --- | --- |
| Input shape | `(1, 224, 224, 3)` |
| Preprocessing | `tensorflow.keras.applications.efficientnet.preprocess_input` |
| Output shape | `(1, 7)`, softmax |

If the number of outputs does not match the number of labels, `/predict` returns a
clear error rather than a wrong label.
