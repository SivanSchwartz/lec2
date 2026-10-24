from pytorch_lightning.callbacks import EarlyStopping, Callback, TQDMProgressBar


class ColabProgressBar(TQDMProgressBar):
    """tqdm bar without the separate validation bar.

    The validation bar is drawn on a second line, and Colab can't redraw a second
    line in place, so every update would be printed as a new line.
    Validation metrics still appear at the end of the main epoch bar.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class EpochMetricsCallback(Callback):
    def _log_metrics(self, trainer, names):
        metrics = {
            f"epoch/{name}": trainer.callback_metrics[name]
            for name in names
            if name in trainer.callback_metrics
        }

        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(
            trainer,
            ["train_loss", "train_accuracy", "train_f1_score"],
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ["val_loss"])

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ["test_loss"])
