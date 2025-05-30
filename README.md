# noisy-graph-classification

Repository of the team "OopsAllNoise" (members: Matteo Pannacci 1948942, Emiliano Paradiso ???) for the Exam Hackaton "[GraphClassificationNoisyLabels](https://huggingface.co/spaces/examhackaton/GraphClassificationNoisyLabels)" done for the course of "Deep Learning" during the MSc in Artificial Intelligence and Robotics at Sapienza University of Rome, A.Y. 2024-2025.

## Command-line Arguments

    # General
    --train_path: type=str, help='Path to the training dataset (optional).'
    --test_path: type=str, required=True, help='Path to the test dataset.'
    --val_proportion: type=float, default=0.0, help='proportion of the train set to use for the validation set'
    --num_checkpoints: type=int, help='Number of checkpoints to save during training.'
    --device: type=int, default=1, help='which gpu to use if any (default: 0)'
    --seed: type=int, default=42, help='random seed'
    --save_all_best: type=bool, default=False
    --from_pretrain: type=bool, default=False
    --predict_with_ensemble: type=bool, default=True

    # Architecture
    --gnn_type: type=str, default='gin', choices=['gin', 'gcn', 'gat'], help='GNN type: gin or gcn'
    --virtual_node: type=bool, default=True, help='Use virtual node or not'
    --residual: type=bool, default=False, help='Using residual connection or not'
    --drop_ratio: type=float, default=0.5, help='dropout ratio (default: 0.5)'
    --num_layer: type=int, default=5, help='number of GNN message passing layers (default: 5)'
    --emb_dim: type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)'
    --graph_pooling_type: type=str, default='mean', help='mean, sum, max, attention, set2set'
    --jk: type=str, default='last', choices=['last', 'sum']
    --aggr_type: type=str, default='add', choices=['add', 'mean']

    # Training
    --lr: type=float, default=0.001, help='optimizer learning rate'
    --batch_size: type=int, default=32, help='input batch size for training (default: 32)'
    --epochs: type=int, default=10, help='number of epochs to train (default: 10)'
    --train_from_best: type=bool, default=False
    --optimizer_type: type=str, default='adam', choices=['adam','adamw']

    # Loss
    --loss_type: type=int, default=1, help='[1]: CrossEntropy; [2]: NoisyCrossEntropy; [3] SymmetricCrossEntropy; [4] NCOD; [5] GeneralizedCrossEntropy; [6] NoisyCrossEntropyCustom'
    --noise_prob: type=float, default=0.2
    --weight_decay: type=float, default=0.0
    --use_class_weights: type=bool, default=False, help='use class weights in the loss computation'
    --q: type=float, default=0.5
    --alpha: type=float, default=1.0
    --beta: type=float, default=1.0
    --label_smoothing: type=float, default=0.0
