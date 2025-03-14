from utils.cl_utils.cl_utils import return_components, create_strategy, run_strategy
from utils.utils import update_perf

DATASETS = [f"weather"]
ROOT = (
    "datasets"
)
MB_SIZE = 50
SUFFIX = ""
STRATEGIES = ["naive", "er", "er_lwf", "agem", "lwf", "ewc", "mir"]

import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    DataAttribute,
    as_taskaware_classification_dataset,
)
from avalanche.benchmarks import benchmark_from_datasets, with_task_labels
from avalanche.benchmarks.scenarios.online import split_online_stream
import os
import pandas as pd
from utils.utils import ROLLING_WINDOWS

if SUFFIX is not None and SUFFIX != "":
    SUFFIX = "_" + SUFFIX
for DATASET in DATASETS:
    print(DATASET)
    INPUT_SIZE = 4
    experiences = {"train": [], "test": []}
    strategy_kwargs = {
        "mem_size": 500,
        "sample_size": 200,
        "alpha": 1,
        "temperature": 2,
        "batch_size_mem": MB_SIZE,
        "freeze_remaining_model": True,
    }

    for eval_ in ("train", "test"):
        df = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_{eval_}.csv"))
        n_exp = len(df["concept"].unique())
        for i in range(n_exp):
            df_task = df[df["concept"] == i + 1].drop(columns="concept")
            x_data = torch.Tensor(df_task.iloc[:, :-1].values)
            y_data = torch.Tensor(df_task.iloc[:, -1].values).to(torch.int)
            torch_data = TensorDataset(x_data)
            task_labels = torch.ones(len(x_data), dtype=torch.int8) * i
            # TODO: build sequences
            # For each data point you must build a sequence containing the previous 10 data points' features and
            # the current data point's feature. It should be a tensor (concept_len, 11, 4)
            # The first 10 data points will have a None/random prediction since it is not possible to build a sequence
            # on it
            avl_data = AvalancheDataset(
                datasets=torch_data,
                data_attributes=[
                    DataAttribute(y_data, name="targets", use_in_getitem=True),
                    DataAttribute(
                        task_labels, name="targets_task_labels", use_in_getitem=True
                    ),
                ],
            )
            avl_data = as_taskaware_classification_dataset(avl_data)
            experiences[eval_].append(avl_data)

    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_test.csv"))
    X_test = []
    y_test = []
    for t in range(n_exp):
        df_task = df[df["concept"] == t + 1].drop(columns="concept")
        X_test.append(torch.Tensor(df_task.iloc[:, :-1].values))
        y_test.append(df_task.iloc[:, -1].values)

    bm = with_task_labels(
        benchmark_from_datasets(train=experiences["train"], test=experiences["test"])
    )

    for exp in bm.train_stream:
        exp.task_labels = list(exp.task_labels)
    for exp in bm.test_stream:
        exp.task_labels = list(exp.task_labels)

    online_train_stream = split_online_stream(
        bm.train_stream, experience_size=MB_SIZE, shuffle=False
    )

    perf = {}
    perf_values = {}
    predictions = {}
    cl_table = {}

    for strategy in STRATEGIES:
        print(strategy)

        components = return_components(strategy, input_size=INPUT_SIZE)
        perf, perf_values, predictions, cl_table = update_perf(
            perf, perf_values, predictions, cl_table, strategy, ROLLING_WINDOWS
        )

        strategy_kwargs["model"] = components["model"]
        cl_strategy = create_strategy(
            name=strategy,
            components=components,
            mb_size=MB_SIZE,
            strategy_kwargs=strategy_kwargs,
        )

        perf, perf_values, predictions, cl_table = run_strategy(
            cl_strategy,
            components["model"],
            strategy,
            ROOT,
            DATASET,
            online_train_stream,
            X_test,
            y_test,
            perf,
            perf_values,
            predictions,
            cl_table,
            SUFFIX,
            ROLLING_WINDOWS
        )
        print()
