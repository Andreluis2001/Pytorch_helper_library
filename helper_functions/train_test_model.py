import torch
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict, List

def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.nn.Module
) -> Tuple[float, float]:

  model.train()

  train_loss, train_acc= 0, 0

  for X, y in train_dataloader:

    train_pred = model(X)

    loss = loss_fn(train_pred, y)
    train_loss += loss.item()

    accuracy = accuracy_score(
        y,
        torch.argmax(torch.softmax(train_pred, dim=1), dim=1)
    )
    train_acc += accuracy

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss, train_acc

def test_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module
) -> Tuple[float, float]:

  test_loss, test_acc= 0, 0

  model.eval()

  with torch.inference_mode():

    for X, y in test_dataloader:

      test_pred = model(X)

      loss = loss_fn(test_pred, y)
      test_loss += loss.item()

      accuracy = accuracy_score(
          y,
          torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
      )
      test_acc += accuracy

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  return test_loss, test_acc

def train_test_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.nn.Module,
    epochs: int = 5
) -> Dict[str, float]:

  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in range(epochs):

    train_loss, train_acc = train_model(model, train_dataloader, loss_fn, optimizer)
    test_loss, test_acc = test_model(model, test_dataloader, loss_fn)

    print(
        "-------------------------------------\n"
        f"Epoch: {epoch+1} |\n"
        f"Train Loss: {train_loss:.4f} |\n"
        f"Train Acc: {(train_acc * 100):.4f}% |\n"
        f"Test Loss: {test_loss:.4f} |\n"
        f"Test Acc: {(test_acc * 100):.4f}%\n"
        "-------------------------------------\n"
    )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
