"""
python train_from_scratch.py --config-path='config' --config-name='finetuned_model_closed_task.yaml' \
    name="some_experiment_name" \
    exp_manager.name="some_experiment_name"
    
"""
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.utils import logging
from nemo.core.config import hydra_runner
from label_models import EncDecSpeakerLabelModel
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf", config_name="finetuned_model_closed_task.yaml")
def main(cfg):
    
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    _ = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    
    try:
        ce_weight = torch.clone(speaker_model.loss.weight)
    except:
        pass
    
    speaker_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    print(speaker_model.loss)
    
    try:
        speaker_model.loss.weight = ce_weight
        print("Cross Entropy Weight:", ce_weight)
    except:
        pass
    
    trainer.fit(speaker_model)

    torch.distributed.destroy_process_group()
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if trainer.is_global_zero:
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator, strategy=cfg.trainer.strategy)
            if speaker_model.prepare_test(trainer):
                trainer.test(speaker_model)


if __name__ == '__main__':
    main()
