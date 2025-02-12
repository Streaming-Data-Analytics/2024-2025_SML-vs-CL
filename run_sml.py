from utils.sml_utils.temporally_augmented_classifier import TemporallyAugmentedClassifier

DATASETS = [f"weather.csv"]
ROOT = (
    "/datasets"
)
SUFFIX = ""

import pickle
import pandas as pd
from river import stream
from river import tree
from utils.sml_utils.sml_utils import test_cl, create_arf
import os
from utils.utils import return_metrics, make_dir, return_rolling
from utils.utils import ROLLING_WINDOWS

if SUFFIX is not None and SUFFIX != "":
    SUFFIX = "_" + SUFFIX
for DATASET in DATASETS:
    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_train.csv"), nrows=1)
    last_task = df["concept"].iloc[0]
    converters = {c: float for c in df.columns if "feat" in c}
    converters["target"] = int
    converters["concept"] = int

    data_stream = stream.iter_csv(
        os.path.join(ROOT, "datasets", f"{DATASET}_train.csv"),
        converters=converters,
        target="target",
    )

    models = {
        "arf": create_arf(),
        "arf_ta": TemporallyAugmentedClassifier(
            base_learner=create_arf(),
            num_old_labels=10,
        )
    }

    perf = {m: {"total": return_metrics(), "concept": return_metrics()} for m in models}

    for m in models:
        for window in ROLLING_WINDOWS:
            perf[m][f"rolling_{window}"] = return_rolling(window)
            perf[m][f"rolling_reset_{window}"] = return_rolling(window)
    perf_values = {
        m: {
            "total": {"accuracy": [], "kappa": []},
            "concept": {"accuracy": [], "kappa": []},
        }
        for m in models
    }
    for m in models:
        for window in ROLLING_WINDOWS:
            perf_values[m][f"rolling_{window}"] = {"accuracy": [], "kappa": []}
            perf_values[m][f"rolling_reset_{window}"] = {"accuracy": [], "kappa": []}
    perf_values["drifts"] = []

    predictions = {m: [] for m in models}

    cl_table = {
        m: {metric: [] for metric in ["accuracy", "kappa"]}
        for m in models.keys()
    }

    df_test = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_test.csv"))

    X_test = []
    y_test = []
    for task in df_test["concept"].unique():
        df_task = df_test[df_test["concept"] == task]
        X_test.append(df_task.iloc[:, :-2].values)
        y_test.append(list(df_task["target"]))

    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_train.csv"))

    for idx, (x, y) in enumerate(data_stream):
        print(f"{DATASET} Prequential {idx+1}", end="\r")
        if last_task != x["concept"]:
            print()
            print(f"DRIFT {idx+1}")
            for m in perf:
                perf[m]["concept"] = return_metrics()
                for window in ROLLING_WINDOWS:
                    perf[m][f"rolling_{window}"] = return_rolling(window)
                    perf[m][f"rolling_reset_{window}"] = return_rolling(window)
            cl_table = test_cl(cl_table, models, X_test, y_test)
            perf_values["drifts"].append(idx)
        last_task = x["concept"]
        del x["concept"]
        for m in models:
            pred = models[m].predict_one(x)
            pred = 0 if pred is None else pred
            predictions[m].append(pred)
            for method in perf[m]:
                if method == "drifts":
                    continue
                for metric in perf[m][method]:
                    perf[m][method][metric].update(y, pred)
                    perf_values[m][method][metric].append(perf[m][method][metric].get())
            models[m].learn_one(x, y)
    cl_table = test_cl(cl_table, models, X_test, y_test)
    make_dir(os.path.join(ROOT, "performance", DATASET))
    with open(
        os.path.join(ROOT, "performance", DATASET, f"performance_sml{SUFFIX}.pkl"), "wb"
    ) as f:
        pickle.dump(perf_values, f)
    with open(
        os.path.join(ROOT, "performance", DATASET, f"cl_table_sml{SUFFIX}.pkl"), "wb"
    ) as f:
        pickle.dump(cl_table, f)
    with open(
        os.path.join(ROOT, "performance", DATASET, f"predictions_sml{SUFFIX}.pkl"), "wb"
    ) as f:
        pickle.dump(predictions, f)

