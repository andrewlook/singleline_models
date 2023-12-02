import torch

from sketch_rnn import HParams, Trainer

if __name__ == "__main__":
    hp = HParams()
    hp.learning_rate = 1e-3
    trainer = Trainer(hp=hp, use_wandb=False,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()