from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torch.distributed as dist
import utils
import numpy as np

import logging

logger = logging.getLogger(__name__)


class LoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.logger_nest = logging.getLogger("exp_logger")
        self.val_outputs = []
        self.learning_rates = []
    
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.local_rank == 0:
            self.val_outputs = []
        return super().on_validation_epoch_start(trainer, pl_module)
    
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if trainer.local_rank == 0:
            self.val_outputs.append(outputs)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.local_rank == 0:
            all_losses_meter = {
                'total_loss': utils.AvgrageMeter(),
            }
            for output in self.val_outputs:
                n = output['nsamples']
                for k, v in output['losses'].items():
                    if k not in all_losses_meter:
                        all_losses_meter[k] = utils.AvgrageMeter()
                    all_losses_meter[k].update(v, n)
                all_losses_meter['total_loss'].update(output['total_loss'], n)
            
            all_losses_meter = {k: round(v.avg, 4) for k, v in all_losses_meter.items()}
                
            # print(f"\n==============\n "
            #     f"valid results: {all_losses_meter}"
            #     f"\n==============\n")
            
            msg = "Evaluation results at epoch {} step {}: ".format(trainer.current_epoch, pl_module.global_step)
            for k, v in all_losses_meter.items():
                msg += "{}: {:.4f}, ".format(k, v)
            
            self.logger_nest.info(msg)
        
            self.val_outputs.clear()
        return super().on_validation_epoch_end(trainer, pl_module)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.local_rank == 0:
            should_log_metrics = batch_idx % (trainer.log_every_n_steps * trainer.accumulate_grad_batches) == 0
            
            
            if should_log_metrics:
                
                msg = "[Epoch {}/{}, step {}/{}] ".format(trainer.current_epoch, trainer.min_epochs, pl_module.global_step, trainer.num_training_batches // trainer.accumulate_grad_batches)
                
                for k, v in outputs["progress_bar"].items():
                    if k == "batch_size":
                        msg += "{}: {:.4f}, ".format(k.replace("tr/", ""), v * trainer.accumulate_grad_batches)
                    else:
                        msg += "{}: {:.4f}, ".format(k.replace("tr/", ""), v)
                msg += "lr: {:.6f}".format(np.mean(self.learning_rates))
                self.learning_rates.clear()
                self.logger_nest.info(msg)
        
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer, opt_idx=0) -> None:
        self.learning_rates.append(optimizer.param_groups[0]['lr'])
        return super().on_before_optimizer_step(trainer, pl_module, optimizer)
