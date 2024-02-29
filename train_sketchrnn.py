import torch

from singleline_models.sketch_rnn.trainer import SketchRNNModel, Trainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hp = SketchRNNModel.get_default_config()
    hp.learning_rate = 1e-3

    model = SketchRNNModel(hp, device=device)

    trainer = Trainer(model=model, hp=hp, use_wandb=False, device=device)
    trainer.train()