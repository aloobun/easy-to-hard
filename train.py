"""
Train a rnn for solving chess puzzles,
based on the paper "Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks".
"""

import argparse
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import wandb


PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}
NUM_PIECES = 12
BOARD_SIZE = 8


def fen_to_tensor(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    tensor = torch.zeros(NUM_PIECES, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, BOARD_SIZE)
            channel = PIECE_TO_CHANNEL[piece.symbol()]
            tensor[channel, row, col] = 1
    return tensor


def uci_move_to_tensor(uci_move: str) -> torch.Tensor: 
    """Convert UCI move to binary tensor with 1s at from/to squares"""
    move = chess.Move.from_uci(uci_move)
    tensor = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    from_row, from_col = divmod(move.from_square, BOARD_SIZE)
    to_row, to_col = divmod(move.to_square, BOARD_SIZE)
    tensor[from_row, from_col] = 1
    tensor[to_row, to_col] = 1
    return tensor


class LichessPuzzlesDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        fen = item['FEN']
        # the 'Moves' field can have multiple moves and for now we only use the first one
        first_move_uci = item['Moves'].split(' ')[0]

        input_tensor = fen_to_tensor(fen)
        target_tensor = uci_move_to_tensor(first_move_uci)

        return input_tensor, target_tensor


class RecurrentBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class RecurrentChessModel(nn.Module):
    def __init__(self, in_channels=12, recurrent_channels=256, out_channels=1):
        super().__init__()
        # encoder: maps input from 12 channels to the recurrent channel dimension
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, recurrent_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(recurrent_channels),
            nn.ReLU(inplace=True)
        )
        
        # single, shared recurrent block
        self.recurrent_block = RecurrentBlock(recurrent_channels)
        
        # head: maps from the recurrent channel dimension to the output (single channel for move probability)
        self.head = nn.Sequential(
            nn.Conv2d(recurrent_channels, out_channels, kernel_size=1),
            nn.Sigmoid()  # Output probabilities directly
        )

    def forward(self, x, num_iterations):
        x = self.encoder(x)
        
        outputs = []
        for _ in range(num_iterations):
            x = self.recurrent_block(x)
            # We compute the output at each step
            output = self.head(x)
            outputs.append(output)
            
        return outputs


def train_one_epoch(model, dataloader, optimizer, criterion, device, num_iterations, scaler, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Debug
        if epoch == 0 and batch_idx == 0:
            print(f"\nDEBUG - Target tensor stats:")
            print(f"Shape: {targets.shape}")
            print(f"Unique values: {torch.unique(targets)}")
            print(f"Zeros: {(targets == 0).sum().item()}, Ones: {(targets == 1).sum().item()}")
            print(f"Ratio of ones: {(targets == 1).float().mean().item():.4f}")
        
        optimizer.zero_grad()

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs, num_iterations)
            iteration_losses = []
            for out in outputs:
                # Squeeze to remove channel dimension: (B, 1, 8, 8) -> (B, 8, 8)
                out = out.squeeze(1)
                loss = criterion(out, targets)
                iteration_losses.append(loss)
            loss = torch.stack(iteration_losses).mean()

        if epoch == 0 and batch_idx < 3:
            with torch.no_grad():
                first_output = outputs[0].squeeze(1)
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                print(f"  Output mean: {first_output.mean().item():.4f}")
                print(f"  Target mean: {targets.float().mean().item():.4f}")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
        if hasattr(args, 'use_wandb') and args.use_wandb:
            wandb.log({
                "step_loss": loss.item(),
                "step": epoch * len(dataloader) + batch_idx
            })

    return total_loss / len(dataloader)
    

def evaluate(model, dataloader, device, max_iterations):
    model.eval()
    correct = 0
    total = 0
    
    use_amp = (device.type == 'cuda')
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                all_outputs = model(inputs, max_iterations)
            
            for i in range(inputs.size(0)):
                # Get outputs for this item across all iterations
                item_outputs = [out[i].squeeze(0) for out in all_outputs]  # Remove channel dim
                item_target = targets[i]
                
                # Find iteration with highest confidence on target positions
                confidences = []
                for out in item_outputs:
                    # Calculate mean prediction on target squares
                    target_mask = (item_target == 1)
                    if target_mask.sum() > 0:
                        confidence = out[target_mask].mean()
                        confidences.append(confidence.cpu().item())
                    else:
                        confidences.append(0.0)
                
                best_iter_idx = np.argmax(confidences)
                best_output = item_outputs[best_iter_idx]

                # Get prediction: find the 2 squares with highest probability
                _, pred_indices = torch.topk(best_output.flatten(), 2)
                
                # Get true indices
                true_indices = item_target.flatten().nonzero().flatten()
                
                # Check if prediction matches target (same 2 squares, order doesn't matter)
                if len(true_indices) == 2 and torch.all(torch.sort(pred_indices)[0] == torch.sort(true_indices)[0]):
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a recurrent model for chess puzzles.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--train-iterations", type=int, default=10, help="Recurrent iterations for training.")
    parser.add_argument("--eval-iterations", type=int, default=20, help="Max recurrent iterations for evaluation.")
    parser.add_argument("--limit-data", type=int, default=None, help="Limit dataset size for quick testing.")
    
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the best model.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable logging with Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="chess-recurrent-model", help="WandB project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity (username or team).")
    
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using: {device}")

    print("Loading dataset")
    full_dataset = load_dataset("Lichess/chess-puzzles", split='train', streaming=False)
    
    print("Performing a 90/10 random split of the dataset")
    split_dataset = full_dataset.shuffle(seed=42).train_test_split(test_size=0.1)
    train_data_hf = split_dataset['train']
    test_data_hf = split_dataset['test']
    
    if args.limit_data:
        train_limit = args.limit_data
        test_limit = int(train_limit * 0.2)
        print(f"Limiting data to {train_limit} train and {test_limit} test samples.")
        train_data_hf = train_data_hf.select(range(train_limit))
        test_data_hf = test_data_hf.select(range(min(test_limit, len(test_data_hf))))

    print(f"{len(train_data_hf)} training examples and {len(test_data_hf)} test examples.")

    train_dataset = LichessPuzzlesDataset(train_data_hf)
    test_dataset = LichessPuzzlesDataset(test_data_hf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Smaller model with single output channel
    model = RecurrentChessModel(recurrent_channels=256, out_channels=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Use Binary Cross Entropy Loss with class weights
    pos_weight = torch.tensor([31.0]).to(device)  # Weight for positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Model init. Starting training for {args.epochs} epochs.")
    print(f"Batch Size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Train Iterations: {args.train_iterations}, Eval Iterations: {args.eval_iterations}")

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    best_accuracy = 0.0

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args
        )

    scaler = torch.cuda.amp.GradScaler() if use_cuda else None

    for epoch in range(args.epochs):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.train_iterations, scaler, epoch
        )
        
        accuracy = evaluate(model, test_loader, device, args.eval_iterations)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.4f}")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_accuracy": accuracy,
                "best_accuracy": max(best_accuracy, accuracy)
            })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"model saved to {model_path} with accuracy: {accuracy:.4f}")
            if args.use_wandb:
                wandb.run.summary["best_accuracy"] = best_accuracy
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.2e}")
        
        if args.use_wandb:
            wandb.log({"learning_rate": current_lr})

    print("Training done")
    if args.use_wandb:
        wandb.finish()
