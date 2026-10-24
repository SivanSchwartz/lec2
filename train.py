import torch
import pytorch_lightning as pl
from model import NN
from datamodule import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium") # to make lightning happy


def run(mode):
    """Train and evaluate one experiment. mode: see config.MODES."""
    pl.seed_everything(42)  # same initialization and data order -> fair comparison

    logger = TensorBoardLogger("tb_logs", name=mode)

    model = NN(
        learning_rate=config.FINETUNE_LEARNING_RATE if mode == "finetune" else config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
        freeze_encoder=(mode == "linear_probe"),
    )

    if mode in ["linear_probe", "finetune"]:
        # Transfer: start from the encoder learned on MNIST in step 1.
        # The MNIST head is not loaded -> the new head starts from random weights.
        model.encoder.load_state_dict(torch.load(config.ENCODER_PATH, map_location="cpu"))  # works for files saved on GPU or CPU

    if mode == "pretrain":
        dataset_name, n_train, num_epochs = config.SOURCE_DATASET, None, config.NUM_EPOCHS
    else:
        dataset_name, n_train, num_epochs = config.TARGET_DATASET, config.N_TARGET_TRAIN, config.TARGET_NUM_EPOCHS

    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        dataset_name=dataset_name,
        n_train=n_train,
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=num_epochs,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss"), TQDMProgressBar(refresh_rate=20)],  # tqdm updates in place on Colab (Rich does not)
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    results = trainer.test(model, dm)

    if mode == "pretrain":
        # Keep the encoder f_theta, throw away the MNIST head h_phi
        torch.save(model.encoder.state_dict(), config.ENCODER_PATH)

    return results[0]["test_accuracy"]


if __name__ == "__main__":
    test_accuracy = {}
    for mode in config.MODES:
        test_accuracy[mode] = run(mode)

    print("\nTest accuracy:")
    for mode, acc in test_accuracy.items():
        dataset_name = config.SOURCE_DATASET if mode == "pretrain" else config.TARGET_DATASET
        print(f"  {mode:<13} ({dataset_name}): {acc:.3f}")
