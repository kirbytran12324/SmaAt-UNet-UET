import argparse
import os
import gc
import json

# Reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import optuna
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from models import unet_precip_regression_lightning as unet_regr
from root import ROOT_DIR

BEST_TRIAL_PATH = "best_trial.json"

def objective(trial):
    parser = argparse.ArgumentParser()
    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)

    # Default args
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args(args=[])

    # Fixed values
    args.n_channels = 12
    args.num_input_images = 12
    args.num_output_images = 6
    args.n_classes = 1
    args.gpus = 1
    args.model = "UNetDS_Attention"
    args.es_patience = 15
    args.lr_patience = 4
    args.reduction_ratio = 16
    args.dataset_folder = str(ROOT_DIR / "Radar" / "dataset" / "train_test_input-length_12_image-ahead_6_rain-threshold_0.h5")
    args.use_oversampled_dataset = True

    # Hyperparameters from Optuna
    args.batch_size = trial.suggest_categorical("batch_size", [4, 6, 8, 12])
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    args.kernels_per_layer = trial.suggest_int("kernels_per_layer", 2, 8)

    model = unet_regr.UNetDS_Attention(hparams=args)

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision=16,
        accumulate_grad_batches=2,
        enable_checkpointing=False,
        logger=False,
        callbacks=[early_stop, pruning_cb],
    )

    trainer.fit(model)
    val = trainer.callback_metrics.get("val_loss")

    # Cleanup
    trainer.strategy.teardown()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return val.item() if val is not None else float("inf")


if __name__ == "__main__":
    # Check if we can resume from best_trial.json
    if os.path.exists(BEST_TRIAL_PATH):
        with open(BEST_TRIAL_PATH, "r") as f:
            best_params = json.load(f)
        print("Resuming from previous best_trial.json")
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.enqueue_trial(best_params)  # Avoids re-running the same trial
    else:
        best_params = None
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    study.optimize(objective, n_trials=20, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best trial params
    with open(BEST_TRIAL_PATH, "w") as f:
        json.dump(trial.params, f, indent=4)
        print(f"Saved best trial params to {BEST_TRIAL_PATH}")
