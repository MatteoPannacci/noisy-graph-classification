# noisy-graph-classification

Repository of the team "OopsAllNoise" (members: Matteo Pannacci 1948942, Emiliano Paradiso 1940454) for the Exam Hackaton "[GraphClassificationNoisyLabels](https://huggingface.co/spaces/examhackaton/GraphClassificationNoisyLabels)" done for the course of "Deep Learning" during the MSc in Artificial Intelligence and Robotics at Sapienza University of Rome, A.Y. 2024-2025.

## Command-line Arguments

    # General
    --train_path (str): Path to the training dataset (optional).
    --test_path (str): Path to the test dataset.
    --val_proportion (float, 0.0): proportion of the train set to use for the validation set
    --num_checkpoints (int): Number of checkpoints to save during training.
    --device (int, 1): which gpu to use if any
    --seed (int, 42): random seed
    --save_all_best (bool, False):
    --from_pretrain (bool, False):
    --predict_with_ensemble (bool, True):

    # Architecture
    --gnn_type (str, 'gin'): GNN type: gin or gcn
    --virtual_node (bool, True): Use virtual node or not
    --residual (bool, False): Using residual connection or not
    --drop_ratio (float, 0.5): dropout ratio
    --num_layer (int, 5): number of GNN message passing layers
    --emb_dim (int, 300): dimensionality of hidden units in GNNs
    --graph_pooling_type (str, 'mean'): mean, sum, max, attention, set2set
    --jk (str, 'last'): last, sum
    --aggr_type (str, 'add'): add, mean

    # Training
    --lr (float, 0.001): optimizer learning rate
    --batch_size (int, 32): input batch size for training
    --epochs (int, 10): number of epochs to train
    --optimizer_type (str, 'adam'): adam, adamw

    # Loss
    --loss_type (int, 1): [1]: CrossEntropy; [2]: NoisyCrossEntropy; [3] SymmetricCrossEntropy; [4] NCOD; [5] GeneralizedCrossEntropy; [6] NoisyCrossEntropyCustom
    --noise_prob (float, 0.2):
    --weight_decay (float, 0.0):
    --use_class_weights (bool, False): use class weights in the loss computation
    --q (float, 0.5):
    --alpha (float, 1.0):
    --beta (float, 1.0):
    --label_smoothing (float, 0.0):
