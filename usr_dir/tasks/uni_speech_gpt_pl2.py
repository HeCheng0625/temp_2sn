import matplotlib

matplotlib.use('Agg')


from usr_dir.datasets.datamodule_pl2 import GPTTtsDataModule

from .pl2_task_base import TTSPL2TaskBase


import logging

logger = logging.getLogger(__name__)


class GPTTtsTASK(TTSPL2TaskBase):
    def __init__(self) -> None:
        super().__init__()
        
    def initialize_data_module(self):
        return GPTTtsDataModule(use_ddp=True) 
    
    def initialize_model(self):
        from usr_dir.module.gpt_tts_pl2 import TTSLatentGPT
        return TTSLatentGPT()

if __name__ == '__main__':
    task = GPTTtsTASK()
