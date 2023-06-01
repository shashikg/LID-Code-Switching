"""
python train_from_scratch.py --config-path='config' --config-name='base_model_closed_task.yaml' \
    name="some_experiment_name" \
    exp_manager.name="some_experiment_name"
"""

import os, torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.utils import logging
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from label_models import EncDecSpeakerLabelModel

@hydra_runner(config_path="configs", config_name="base_model_closed_task.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(speaker_model)
    if not trainer.fast_dev_run:
        model_path = os.path.join(log_dir, '..', 'spkr.nemo')
        speaker_model.save_to(model_path)

    torch.distributed.destroy_process_group()
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if trainer.is_global_zero:
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator, strategy=cfg.trainer.strategy)
            if speaker_model.prepare_test(trainer):
                trainer.test(speaker_model)


if __name__ == '__main__':
    main()
