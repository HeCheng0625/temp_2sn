# ---------------------------------------------
#              Task Base class
#----------------------------------------------

import os
import torch
from utils.hparams import hparams, set_hparams
from utils.pl_utils_2 import LoggingCallback

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.strategies as pl_strategies
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)


class TTSPL2TaskBase:
    def __init__(self) -> None:
        set_hparams()
        
        print('| set hparams: ')
        for i, (k, v) in enumerate(sorted(hparams.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        
        call_back_lists = self.initialize_callbacks()
        custermized_logger = self.initialize_logger()
        strategy = self.initialize_strategy()
        
        
        self.pl_trainer = Trainer(
            callbacks=call_back_lists,
            use_distributed_sampler=False,
            #strategy=pl_strategies.DDPStrategy(find_unused_parameters=True),
            strategy=strategy,
            accumulate_grad_batches=hparams["accumulate_grad_batches"],
            gradient_clip_val=hparams['clip_grad_norm'],
            val_check_interval=hparams['val_check_interval'] * hparams["accumulate_grad_batches"],
            enable_progress_bar=False,
            logger=custermized_logger,
            log_every_n_steps=hparams['log_interval'],
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams[
                                  'validate'] else 10000,
            precision=hparams["precision"],
            default_root_dir=hparams['work_dir'],
            num_nodes=hparams['num_nodes'], 
            check_val_every_n_epoch=None,
        )

        self.model = self.initialize_model()
        
        self.data_module = self.initialize_data_module()
        # self.model = torch.compile(self.model)
        
        if not hparams['infer']:  # train
            
            # hard coding
            # ckpt_path = os.path.join(hparams['work_dir'], "0-16300.ckpt")
            ckpts = []
            for fn in os.listdir(hparams['work_dir']):
                if fn.endswith(".ckpt"):
                    ckpts.append(fn)
            if len(ckpts):
                # epoch-updates.ckpt
                ckpts.sort(key=lambda x: (int(x.split("-")[0]), int(x.replace('.ckpt', '').split("-")[1])), reverse=True)
                ckpt_path = os.path.join(hparams['work_dir'], ckpts[0])
            else:
                ckpt_path = "last"
            # print(ckpt_path)
            # exit()
            self.pl_trainer.fit(model=self.model, datamodule=self.data_module, ckpt_path=ckpt_path)
        else:
            ckpt_path = os.path.join(hparams['work_dir'], hparams["inference_ckpt"])
            if not os.path.exists(ckpt_path):
                raise ValueError(f"ckpt path {ckpt_path} not exists")
            self.pl_trainer.test(model=self.model, datamodule=self.data_module, ckpt_path=ckpt_path)
    
    @abstractmethod
    def initialize_data_module(self) -> pl.LightningDataModule:
        raise NotImplementedError()
    
    @abstractmethod
    def initialize_model(self) -> pl.LightningModule:
        raise NotImplementedError()
    
    def initialize_callbacks(self):
        checkpoint_callback =  ModelCheckpoint(
                                    dirpath=hparams['work_dir'],
                                    save_top_k=-1,
                                    auto_insert_metric_name=False,
                                    every_n_train_steps=hparams['val_check_interval'],
                                )
        text_logger_cb = LoggingCallback()
        
        learning_rate = LearningRateMonitor(logging_interval='step')
        return [text_logger_cb, checkpoint_callback, learning_rate]
    
    def initialize_logger(self):
        ret_logger = pl_loggers.CometLogger(save_dir="lightning_logs", project_name="large-scale-tts", experiment_name=hparams["exp_name"], offline=True)
        return ret_logger
    
    def initialize_strategy(self):
        if hparams['strategy'] == 'ddp':
            logger.info("------------- Using ddp strategy --------------")
            os.system("sleep 5")
            strategy = pl_strategies.DDPStrategy(find_unused_parameters=True)
        elif hparams['strategy'] == 'deepspeed':
            logger.info("------------- Using deepspeed strategy --------------")
            os.system("sleep 5")
            strategy = pl_strategies.DeepSpeedStrategy(stage=int(hparams["deep_speed_strategy_stage"]))
        elif hparams["strategy"] == "None":
            logger.info("------------- Using None strategy --------------")
            os.system("sleep 5")
            strategy = "auto"
        return strategy
