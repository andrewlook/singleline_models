import torch

from singleline_models.sketch_transformer.trainer import get_default_config, Trainer

if __name__ == "__main__":
    hp = get_default_config()
    hp.learning_rate = 1e-3
    hp.dataset_fname = 'data/look/epoch-20231214-filtered-trainval.npz'
    trainer = Trainer(hp=hp, use_wandb=False,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()