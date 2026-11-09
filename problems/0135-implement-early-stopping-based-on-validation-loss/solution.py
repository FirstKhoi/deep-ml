from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = float('inf')
    best_epoch = 0
    counter = 0

    for epoch, loss in enumerate(val_losses):
        if best_loss - loss > min_delta:
            best_loss = loss
            best_epoch = epoch
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            return epoch, best_epoch

    return best_epoch, epoch