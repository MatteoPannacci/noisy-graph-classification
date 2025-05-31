import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src import *
import pandas as pd
import logging
from tqdm import tqdm
import gc
import torch.nn.utils
from torch.utils.data import random_split
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
import torch.nn.functional as F
from src.models import GNN



def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")



def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):

    model.train()
    total_loss = 0

    f1_metric = F1Score(task="multiclass", num_classes=6, average='macro').to(device)
    accuracy_metric = Accuracy(task="multiclass", num_classes=6).to(device)

    pred_labels = torch.empty(len(data_loader.dataset), device=device)
    true_labels = torch.empty(len(data_loader.dataset), device=device)
    start_idx = 0
    batch_counter = 0

    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):

        # optimization
        data = data.to(device)
        optimizer.zero_grad()
        output, phi = model(data)

        if type(criterion) == ncodLoss:
            loss = criterion(data.id, output, data.y, phi, batch_counter, current_epoch)
        else:
            loss = criterion(output, data.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)

        # store predictions and true labels
        batch_size = data.num_graphs
        end_idx = start_idx + batch_size
        pred_labels[start_idx:end_idx] = pred
        true_labels[start_idx:end_idx] = data.y
        start_idx = end_idx
        batch_counter += 1

    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    f1_score = f1_metric(pred_labels, true_labels).item()
    accuracy = accuracy_metric(pred_labels, true_labels).item()

    return total_loss / len(data_loader), accuracy, f1_score


def evaluate(data_loader, model, device, calculate_accuracy=False, return_labels=False, return_scores=False):

    model.eval()
    
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # use NoisyCrossEntropy?

    f1_metric = F1Score(task="multiclass", num_classes=6, average='macro').to(device)
    accuracy_metric = Accuracy(task="multiclass", num_classes=6).to(device)

    pred_labels = torch.empty(len(data_loader.dataset), device=device, dtype=torch.int64)
    true_labels = torch.empty(len(data_loader.dataset), device=device, dtype=torch.int64)
    scores = torch.empty((len(data_loader.dataset),6), device=device)

    start_idx = 0

    with torch.no_grad():

        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):

            data = data.to(device)
            output, _ = model(data)

            pred = output.argmax(dim=1)

            batch_size = data.num_graphs
            end_idx = start_idx + batch_size
            pred_labels[start_idx:end_idx] = pred
            scores[start_idx:end_idx] = output
            if calculate_accuracy or return_labels:
                total_loss += criterion(output, data.y).item()
                true_labels[start_idx:end_idx] = data.y

            start_idx = end_idx

    if calculate_accuracy:
        f1_score = f1_metric(pred_labels, true_labels).item()
        accuracy = accuracy_metric(pred_labels, true_labels).item()
        return  total_loss / len(data_loader), accuracy, f1_score

    if return_labels:
        return pred_labels.cpu().numpy(), true_labels.cpu().numpy()

    if return_scores:
        return scores.cpu().numpy()

    else:
        return pred_labels.cpu().numpy()



def main(args):

    # Set the random seed
    set_seed(args.seed)

    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

    print("creating model")
    model = GNN(
        gnn_type = args.gnn_type,
        residual = args.residual,
        num_class = 6,
        num_layer = args.num_layer,
        emb_dim = args.emb_dim,
        drop_ratio = args.drop_ratio,
        virtual_node = args.virtual_node,
        graph_pooling = args.graph_pooling_type,
        JK = args.jk,
        aggr_type = args.aggr_type
    ).to(device)

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # Setup logger
    print("setup logging")
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    # Define checkpoint path relative to the script's directory
    print("looking for checkpoints")
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # If train_path is provided, train the model
    if args.train_path:

        if args.from_pretrain:
            print("loading pretrained model")
            pretrain_path = os.path.join(script_dir, "checkpoints", f"model_pretrain_best.pth")
            model.load_state_dict(torch.load(pretrain_path))

        print("loading train datasets")
        use_validation = (args.val_proportion != 0.0)

        if use_validation:

            full_dataset = GraphDataset(args.train_path, transform=add_zeros)
            val_size = int(0.2 * len(full_dataset))
            train_size = len(full_dataset) - val_size

            generator = torch.Generator().manual_seed(12)
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        else:
            train_dataset = GraphDataset(args.train_path, transform=add_zeros)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # compute class weights
        if args.use_class_weights:
            class_weights = (1.0 / torch.tensor(compute_label_distribution(train_loader), dtype=torch.float, device=device))
            class_weights = torch.sqrt(class_weights)
            class_weights = class_weights * len(class_weights) / class_weights.sum()
            print(f"class weights: {class_weights}")
        else:
            class_weights = None

        if args.optimizer_type == 'adam':
            optimizer_type = torch.optim.Adam
        elif args.optimizer_type == 'adamw':
            optimizer_type = torch.optim.AdamW
        else:
            raise ValueError("optimizer not found")

        # choose loss type
        if args.loss_type == 1:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        elif args.loss_type == 2:
            criterion = NoisyCrossEntropyLoss(args.noise_prob, weight=class_weights, label_smoothing=args.label_smoothing)
        elif args.loss_type == 3:
            criterion = SymmetricCrossEntropyLoss(alpha=args.alpha, beta=args.beta)
        elif args.loss_type == 4:

            all_labels = []
            for sample in full_dataset:
                all_labels.append(sample.y)
            all_labels = torch.cat(all_labels)

            criterion = ncodLoss(
                labels = all_labels,
                n = len(full_dataset),
                C = 6,
                ratio_consistency = 0.2,
                ratio_balance = 0.2,
                device = device,
                encoder_features = args.emb_dim,
                total_epochs = args.epochs,
                label_smoothing=args.label_smoothing
            )

        elif args.loss_type == 5:
            criterion = GeneralizedCrossEntropyLoss(q=args.q, weight=class_weights)
        elif args.loss_type == 6:
            criterion = NoisyCrossEntropyLossCustom(args.noise_prob, weight=class_weights, label_smoothing=args.label_smoothing)

        else:
            raise ValueError("criterion not found")

        # setup optimizer
        if args.loss_type == 4:
            optimizer = optimizer_type(
                list(model.parameters()) + list(criterion.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        else:
            optimizer = optimizer_type(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        num_epochs = args.epochs
        best_accuracy = 0.0

        train_losses = []
        train_accuracies = []
        train_f1s = []
        val_losses = []
        val_accuracies = []
        val_f1s = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        print("starting training")
        for epoch in range(num_epochs):

            # train
            train_loss, train_acc, train_f1 = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            # validation
            if use_validation:
                val_loss, val_acc, val_f1 = evaluate(val_loader, model, device, calculate_accuracy=True)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)
            else:
                _, train_acc, train_f1 = evaluate(train_loader, model, device, calculate_accuracy=True)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_f1s.append(train_f1)

            # Save logs for training progress
            logger.info("---")
            logger.info(f"Epoch {epoch + 1:0{len(str(num_epochs))}d}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            if use_validation:
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Save best model
            if (use_validation and val_acc > best_accuracy): # use f1 score instead?
                best_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), os.path.join(checkpoints_folder, f"model_{test_dir_name}_best_at_{epoch}.pth"))
                logger.info(f"Best model updated and saved at {checkpoint_path}")
            elif (not use_validation and train_acc > best_accuracy):
                best_accuracy = train_acc
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), os.path.join(checkpoints_folder, f"model_{test_dir_name}_best_at_{epoch}.pth"))
                logger.info(f"Best model updated and saved at {checkpoint_path}")
            
            logger.info("---")

        # Plot training progress in current directory
        plot_progress("Training", train_losses, train_accuracies, train_f1s, os.path.join(logs_folder, "plotsTrain"))
        if use_validation:
            plot_progress("Validation", val_losses, val_accuracies, val_f1s, os.path.join(logs_folder, "plotsVal"))
            plot_all(train_losses, train_accuracies, train_f1s, val_losses, val_accuracies, val_f1s, os.path.join(logs_folder, "plotsAll"))

        # Load best model
        model.load_state_dict(torch.load(checkpoint_path))

        # Plot confusion matrix
        train_pred, train_true = evaluate(train_loader, model, device, return_labels=True)
        plot_confusion_matrix("Training", train_pred, train_true, logs_folder)
        del train_pred
        del train_true
        if use_validation:
            val_pred, val_true = evaluate(val_loader, model, device, return_labels=True)
            plot_confusion_matrix("Validation", val_pred, val_true, logs_folder)
            del val_pred
            del val_true

        # DELETE TRAIN DATASET VARIABLES
        if use_validation:
            del train_dataset
            del train_loader
            del full_dataset
            del val_dataset
            del val_loader
        else:
            del train_dataset
            del train_loader
        gc.collect()

    # Prepare test dataset and loader
    print("loading test datasets")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Generate predictions for the test set using the best model
    if not args.predict_with_ensemble:
        print("generating prediction")
        model.load_state_dict(torch.load(checkpoint_path))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)

    else:
        print("generating predictions with ensemble")

        # fix type of weights according to dataset
        if test_dir_name in ['A', 'D']:
            print("type of ensemble weights: softmax")
            args.ensemble_weights = 'softmax'
            print(args.ensemble_weights)
        else:
            print("type of ensemble weights: voting")
            args.ensemble_weights = 'no'
            print(args.ensemble_weights)

        total_scores = torch.zeros((len(test_dataset),6))
        ensemble_folder = os.path.join(script_dir, f"checkpoints/{test_dir_name}_ensemble")
        for model_name in os.listdir(ensemble_folder):
            model_path = os.path.join(ensemble_folder, model_name)
            model.load_state_dict(torch.load(model_path))
            if args.ensemble_weights == 'softmax':
                model_scores = evaluate(test_loader, model, device, return_scores=True)
            elif args.ensemble_weights == 'no':
                model_scores = F.one_hot(torch.tensor(evaluate(test_loader, model, device)), num_classes=6)
            else:
                raise ValueError("ensemble weight type not found")
            total_scores += model_scores
        predictions = total_scores.argmax(dim=1)
        save_predictions(predictions, args.test_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    # General
    parser.add_argument('--train_path', type=str, help='Path to the training dataset (optional).')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test dataset.')
    parser.add_argument('--val_proportion', type=float, default=0.2, help='proportion of the train set to use for the validation set')
    parser.add_argument('--num_checkpoints', type=int, default=10, help='Number of checkpoints to save during training.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_all_best', type=bool, default=False, action=argparse.BooleanOptionalAction, help='save the model each time it has the highest accuracy')
    parser.add_argument('--from_pretrain', type=bool, default=False, action=argparse.BooleanOptionalAction, help='start the training from the "model_pretrain_best.pth" in the /checkpoints folder')
    parser.add_argument('--predict_with_ensemble', type=bool, default=True, action=argparse.BooleanOptionalAction, help='predict using the models from the ensemble folder')
    parser.add_argument('--ensemble_weights', type=str, choices=['softmax','no'], default='no')

    # Architecture
    parser.add_argument('--gnn_type', type=str, default='gin', choices=['gin', 'gcn'], help='GNN type: gin or gcn')
    parser.add_argument('--virtual_node', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use virtual node')
    parser.add_argument('--residual', type=bool, default=False, action=argparse.BooleanOptionalAction, help='use residual connection')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--num_layer', type=int, default=4, help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=512, help='dimensionality of hidden units in GNNs')
    parser.add_argument('--graph_pooling_type', type=str, default='mean', choices=['mean', 'sum', 'max', 'attention', 'set2set'], help='type of pooling for the overall graph representation')
    parser.add_argument('--jk', type=str, default='last', choices=['last', 'sum'], help='aggregation type along layers for the node representation')
    parser.add_argument('--aggr_type', type=str, default='add', choices=['add', 'mean'], help='aggregation type for the node message passing')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam','adamw'], help='optimizer name')

    # Loss
    parser.add_argument('--loss_type', type=int, default=2, help='[1]: CrossEntropy; [2]: NoisyCrossEntropy; [3] SymmetricCrossEntropy; [4] NCOD; [5] GeneralizedCrossEntropy; [6] NoisyCrossEntropyCustom')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='parameter p for NoisyCrossEntropy and NoisyCrossEntropyCustom')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay regularization')
    parser.add_argument('--use_class_weights', type=bool, default=False, action=argparse.BooleanOptionalAction, help='use class weights in the loss computation')
    parser.add_argument('--q', type=float, default=0.5, help='q parameter for GeneralizedCrossEntropy')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha parameter for SymmetricCrossEntropy')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter for SymmetricCrossEntropy')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing regularization')

    args = parser.parse_args()

    print("starting main")
    main(args)