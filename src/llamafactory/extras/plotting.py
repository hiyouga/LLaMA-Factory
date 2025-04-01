# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import re
from typing import Any, Dict, List

from transformers.trainer import TRAINER_STATE_NAME

from .logging import get_logger
from .packages import is_matplotlib_available

if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""EMA implementation according to TensorBoard."""
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_loss_plot(trainer_log: List[Dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plots loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if log.get("loss", None):
            steps.append(log["current_steps"])
            losses.append(log["loss"])

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def gen_loss_plot_adaclip(trainer_log: List[Dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plots loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if "Loss" in log:
            matches = re.findall(
                r"[+\-]?(?=\.\d|\d)(?:0|[1-9]\d*)?(?:\.\d*)?(?:\d[eE][+\-]?\d+)|[+-]?\d+\.\d+|[+-]?\d+", log
            )
            current_epoch = int(matches[8])
            current_batch = int(matches[9])
            total_batch = int(matches[10])
            current_steps = (current_epoch) * total_batch + current_batch
            total_steps = (current_epoch + 1) * total_batch
            loss = float(matches[12])
            steps.append(current_steps)
            losses.append(loss)

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def gen_loss_plot_clip(trainer_log: List[Dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plots loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    bx = fig.add_subplot(1, 2, 2)
    steps, losses, access = [], [], []
    for log in trainer_log:
        if "loss" in log:
            matches = re.findall(
                "[+\-]?(?=\.\d|\d)(?:0|[1-9]\d*)?(?:\.\d*)?(?:\d[eE][+\-]?\d+)|[+-]?\d+\.\d+|[+-]?\d+", log
            )
            current_epoch = int(matches[0])
            total_epoch = int(matches[1])
            current_batch = int(matches[2])
            total_batch = int(matches[3])
            current_steps = (current_epoch - 1) * total_batch + current_batch
            loss = float(matches[9])
            acc = float(matches[11])
            steps.append(current_steps)
            losses.append(loss)
            access.append(acc)

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    bx.plot(steps, access, color="#1f77b4", alpha=0.4, label="original")
    bx.plot(steps, smooth(access), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    bx.legend()
    bx.set_xlabel("step")
    bx.set_ylabel("acc")
    return fig


def plot_loss(save_dictionary: str, keys: List[str] = ["loss"]) -> None:
    r"""Plots loss curves and saves the image."""
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)
