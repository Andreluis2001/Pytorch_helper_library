import torch
from pathlib import Path

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
) -> None:

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
    model_name += ".pth"

  model_save_path = target_dir_path / model_name

  print(f"Saving model state dictionary to {model_save_path}")

  torch.save(obj=model.state_dict(), f=model_save_path)

def set_seed(seed_to_set: int = 42):

  torch.manual_seed(seed=seed_to_set)
  torch.cuda.manual_seed(seed=seed_to_set)
