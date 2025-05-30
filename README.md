# noisy-graph-classification

Repository of the team "OopsAllNoise" (members: Matteo Pannacci 1948942, Emiliano Paradiso ???) for the Exam Hackaton "[GraphClassificationNoisyLabels](https://huggingface.co/spaces/examhackaton/GraphClassificationNoisyLabels)" done for the course of "Deep Learning" during the MSc in Artificial Intelligence and Robotics at Sapienza University of Rome, A.Y. 2024-2025.

## Command-line Arguments

    # General
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--val_proportion", type=float, default=0.0, help="proportion of the train set to use for the validation set")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_all_best', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--from_pretrain', type=bool, default=False, action=argparse.BooleanOptionalAction)

    # Architecture
    parser.add_argument('--gnn_type', type=str, default='gin', choices=['gin', 'gcn', 'gat'], help='GNN type: gin or gcn')
    parser.add_argument('--virtual_node', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Use virtual node or not')
    parser.add_argument('--residual', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Using residual connection or not')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--graph_pooling_type', type=str, default='mean', help='mean, sum, max, attention, set2set')
    parser.add_argument('--jk', type=str, default="last", choices=['last', 'sum'])
    parser.add_argument('--aggr_type', type=str, default='add', choices=['add', 'mean'])

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--train_from_best', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam','adamw'])

    # Loss
    parser.add_argument('--loss_type', type=int, default=1, help='[1]: CrossEntropy; [2]: NoisyCrossEntropy; [3] SymmetricCrossEntropy; [4] NCOD; [5] GeneralizedCrossEntropy; [6] NoisyCrossEntropyCustom')
    parser.add_argument('--noise_prob', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_class_weights', type=bool, default=False, action=argparse.BooleanOptionalAction, help='use class weights in the loss computation')
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--predict_with_ensemble', type=bool, default=True, action=argparse.BooleanOptionalAction)
