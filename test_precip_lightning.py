import json
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import lightning.pytorch as pl

from root import ROOT_DIR
from utils import dataset_precip, model_classes


def get_model_loss(model, test_dl, loss="mse", denormalize=True):
    model.eval()
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    else:
        raise ValueError(f"Unknown loss: {loss}")
    factor = 1
    if denormalize:
        factor = 260.0

    with torch.no_grad():
        loss_model = 0.0
        for x, y_true in tqdm(test_dl, leave=False):
            x, y_true = x.to("gpu"), y_true.to("gpu")
            y_true = y_true.squeeze(0)
            y_pred = model(x)
            loss = loss_func(y_pred.squeeze() * factor, y_true * factor, reduction="sum") / y_true.size(0)
            loss_model += loss
        loss_model /= len(test_dl)
    return np.array(loss_model)


def get_persistence_metrics(test_dl, denormalize=True):
    loss_func = nn.functional.mse_loss
    factor = 1
    if denormalize:
        factor = 260.0
    threshold = 0.0
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    loss, loss_denorm = 0.0, 0.0

    for x, y_true in tqdm(test_dl, leave=False):
        y_true = y_true.squeeze(0)
        y_pred = x[:, -1, :]
        loss += loss_func(y_pred.squeeze(), y_true, reduction="sum") / y_true.size(0)
        loss_denorm += loss_func(y_pred.squeeze() * factor, y_true * factor, reduction="sum") / y_true.size(0)
        y_pred_adj = y_pred.squeeze() * factor
        y_true_adj = y_true * factor

        y_pred_mask = y_pred_adj > threshold
        y_true_mask = y_true_adj > threshold

        tn, fp, fn, tp = np.bincount(y_true_mask.view(-1).int() * 2 + y_pred_mask.view(-1).int(), minlength=4)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (
        total_tp + total_tn + total_fp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    csi = total_tp / (total_tp + total_fn + total_fp) if (total_tp + total_fn + total_fp) > 0 else 0
    far = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    loss /= len(test_dl)
    loss_denorm /= len(test_dl)
    return loss, loss_denorm, precision, recall, accuracy, f1, csi, far


def print_persistent_metrics(data_file):
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file, num_input_images=6, num_output_images=1, train=False
    )

    test_dl = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0, pin_memory=True)
    loss, loss_denorm, precision, recall, accuracy, f1, csi, far = get_persistence_metrics(test_dl, denormalize=True)
    print(
        f"Loss Persistence (MSE): {loss}, MSE denormalized: {loss_denorm}, precision: {precision}, "
        f"recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}"
    )
    return loss, loss_denorm


def get_model_losses(model_folder, data_file):
    persistence_loss, persistence_loss_denormalized = print_persistent_metrics(data_file)
    test_losses = {
        "Persistence": [{"MSE": persistence_loss.item(), "MSE_denormalized": persistence_loss_denormalized.item()}]
    }

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file, num_input_images=6, num_output_images=1, train=False
    )

    test_dl = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, pin_memory=True)
    trainer = pl.Trainer(logger=False)

    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = model_classes.get_model_class(model_file)
        loaded_model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
        model_loss = trainer.test(model=loaded_model, dataloaders=[test_dl])

        test_losses[model_name] = model_loss
    return test_losses


def plot_losses(test_losses, loss: str):
    names = list(test_losses.keys())
    values = [v[0][loss] for k, v in test_losses.items()]
    plt.figure()
    plt.bar(names, values)
    plt.xticks(rotation=45)
    plt.xlabel("Models")
    plt.ylabel(f"{loss.upper()} on test set")
    plt.title("Comparison of different models")
    plt.show()


if __name__ == "__main__":
    model_folder = ROOT_DIR / "checkpoints" / "comparison"
    data_file = ROOT_DIR / "Radar" / "dataset" / "train_test_input-length_6_image-ahead_1_rain-threshold_0.h5"

    load = False
    save_file = model_folder / "model_losses_MSE.txt"
    if load:
        with open(save_file) as f_load:
            test_losses = json.load(f_load)
    else:
        test_losses = get_model_losses(model_folder, data_file)
        with open(save_file, "w") as f_write:
            json.dump(test_losses, f_write, indent=4)

    print(test_losses)
    plot_losses(test_losses, "MSE")
