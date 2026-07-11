Fine Tune Model
src/training/fine_tune_model.py • Logs: archive+latest • Runtime: 64m 26s • Exit: n/a
running
Progress: 49.5% • Stage: training phase2
Total ETA: 1h 5m
Started: /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/.venv/bin/python3 /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/src/training/fine_tune_model.py
Using device: cuda
Loading checkpoint from: /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/leaf_disease_checkpoint.pt
Starting Phase 2 (Fine-Tuning Backbone)
Unfrozen ALL layers of the backbone.
Using Exponential Moving Average (EMA) for model weights.
Epoch 6 Train: 100%|██████████| 13782/13782 [26:02<00:00,  8.28it/s, loss=1.0470, acc=0.9879]
Validate: 100%|█████████▉| 1212/1214 [00:37<00:00, 34.38it/s]
Epoch 6/15 - loss: 1.0470 - acc: 0.9879 - val_loss: 1.0726 - val_acc: 0.9646
Saved improved model at epoch 6: acc=0.964571
Epoch 7 Train: 100%|█████████▉| 13781/13782 [25:26<00:00,  9.05it/s, loss=1.1581, acc=0.9569]
Validate: 100%|█████████▉| 1212/1214 [00:36<00:00, 34.51it/s]
Epoch 7/15 - loss: 1.1581 - acc: 0.9569 - val_loss: 1.0900 - val_acc: 0.9604
Epoch 8 Train:  43%|████▎     | 5868/13782 [10:50<14:34,  9.05it/s, loss=1.1375, acc=0.9613]
 Stop
Train Model
src/training/train_model.py • Backbone: DINOv3 • Data/class: 50% • Opt: AdamW • Save: with optimizer • Equalizer: on • Logs: latest-only • Runtime: 210m 47s • Exit: 0
completed
Progress: 100.0% • Stage: training phase2
Total ETA: n/a
Started: /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/.venv/bin/python3 /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/src/training/train_model.py --base-model DINOv3 --train-fraction 0.500000 --optimizer AdamW --save-mode with_optimizer --class-equalizer on --must-review off
Using device: cuda
Using Exponential Moving Average (EMA) for model weights.
Starting Phase 1 (Frozen Backbone)
Epoch 1 Train: 100%|█████████▉| 3445/3446 [04:47<00:00, 13.63it/s, loss=2.2994, acc=0.6596]
Validate: 100%|██████████| 607/607 [00:40<00:00, 14.11it/s]
Epoch 1/5 - loss: 2.2994 - acc: 0.6596 - val_loss: 2.0178 - val_acc: 0.6958
Saved improved model at epoch 1: acc=0.695762
Epoch 2 Train: 100%|██████████| 3446/3446 [04:14<00:00, 14.70it/s, loss=1.8728, acc=0.7712]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.44it/s]
Epoch 2/5 - loss: 1.8728 - acc: 0.7712 - val_loss: 1.7696 - val_acc: 0.7639
Saved improved model at epoch 2: acc=0.763891
Epoch 3 Train: 100%|█████████▉| 3445/3446 [04:14<00:00, 13.75it/s, loss=1.7843, acc=0.7940]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.45it/s]
Epoch 3/5 - loss: 1.7843 - acc: 0.7940 - val_loss: 1.7506 - val_acc: 0.7678
Saved improved model at epoch 3: acc=0.767805
Epoch 4 Train: 100%|█████████▉| 3445/3446 [04:14<00:00, 13.75it/s, loss=1.7241, acc=0.8099]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.46it/s]
Epoch 4/5 - loss: 1.7241 - acc: 0.8099 - val_loss: 1.7199 - val_acc: 0.7851
Saved improved model at epoch 4: acc=0.785056
Epoch 5 Train: 100%|██████████| 3446/3446 [04:14<00:00, 14.86it/s, loss=1.6818, acc=0.8213]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.44it/s]
Epoch 5/5 - loss: 1.6818 - acc: 0.8213 - val_loss: 1.6650 - val_acc: 0.7979
Saved improved model at epoch 5: acc=0.797878
Starting Phase 2 (Fine-Tuning Backbone)
Unfrozen ALL layers of the backbone.
Epoch 6 Train: 100%|██████████| 3446/3446 [11:30<00:00,  5.62it/s, loss=1.6507, acc=0.8285]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.42it/s]
Epoch 6/20 - loss: 1.6507 - acc: 0.8285 - val_loss: 1.6451 - val_acc: 0.8060
Saved improved model at epoch 6: acc=0.805963
Epoch 7 Train: 100%|█████████▉| 3445/3446 [11:29<00:00,  5.02it/s, loss=1.3776, acc=0.9118]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.44it/s]
Epoch 7/20 - loss: 1.3776 - acc: 0.9118 - val_loss: 1.2178 - val_acc: 0.9367
Saved improved model at epoch 7: acc=0.936660
Epoch 8 Train: 100%|█████████▉| 3445/3446 [11:29<00:00,  5.00it/s, loss=1.2509, acc=0.9415]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.47it/s]
Epoch 8/20 - loss: 1.2509 - acc: 0.9415 - val_loss: 1.1557 - val_acc: 0.9504
Saved improved model at epoch 8: acc=0.950409
Epoch 9 Train: 100%|█████████▉| 3445/3446 [11:29<00:00,  5.00it/s, loss=1.1897, acc=0.9547]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.45it/s]
Epoch 9/20 - loss: 1.1897 - acc: 0.9547 - val_loss: 1.1404 - val_acc: 0.9543
Saved improved model at epoch 9: acc=0.954272
Epoch 10 Train: 100%|█████████▉| 3445/3446 [11:29<00:00,  5.01it/s, loss=1.1530, acc=0.9621]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.45it/s]
Epoch 10/20 - loss: 1.1530 - acc: 0.9621 - val_loss: 1.1011 - val_acc: 0.9592
Saved improved model at epoch 10: acc=0.959164
Epoch 11 Train: 100%|█████████▉| 3445/3446 [11:30<00:00,  5.01it/s, loss=1.1249, acc=0.9686]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.45it/s]
Epoch 11/20 - loss: 1.1249 - acc: 0.9686 - val_loss: 1.0950 - val_acc: 0.9622
Saved improved model at epoch 11: acc=0.962150
Epoch 12 Train: 100%|█████████▉| 3445/3446 [11:30<00:00,  5.01it/s, loss=1.1057, acc=0.9727]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.48it/s]
Epoch 12/20 - loss: 1.1057 - acc: 0.9727 - val_loss: 1.0966 - val_acc: 0.9609
Epoch 13 Train: 100%|█████████▉| 3445/3446 [11:29<00:00,  5.00it/s, loss=1.0837, acc=0.9776]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.50it/s]
Epoch 13/20 - loss: 1.0837 - acc: 0.9776 - val_loss: 1.0991 - val_acc: 0.9584
Epoch 14 Train: 100%|█████████▉| 3445/3446 [11:30<00:00,  4.99it/s, loss=1.0697, acc=0.9816]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.43it/s]
Epoch 14/20 - loss: 1.0697 - acc: 0.9816 - val_loss: 1.0948 - val_acc: 0.9602
Epoch 15 Train: 100%|█████████▉| 3445/3446 [11:30<00:00,  5.00it/s, loss=1.0552, acc=0.9851]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.47it/s]
Epoch 15/20 - loss: 1.0552 - acc: 0.9851 - val_loss: 1.0683 - val_acc: 0.9667
Saved improved model at epoch 15: acc=0.966734
Epoch 16 Train: 100%|█████████▉| 3445/3446 [11:30<00:00,  5.00it/s, loss=1.0475, acc=0.9871]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.47it/s]
Epoch 16/20 - loss: 1.0475 - acc: 0.9871 - val_loss: 1.0804 - val_acc: 0.9624
Epoch 17 Train: 100%|█████████▉| 3445/3446 [11:54<00:00,  4.99it/s, loss=1.0384, acc=0.9894]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.46it/s]
Epoch 17/20 - loss: 1.0384 - acc: 0.9894 - val_loss: 1.0997 - val_acc: 0.9597
Epoch 18 Train: 100%|█████████▉| 3445/3446 [11:31<00:00,  5.01it/s, loss=1.0319, acc=0.9911]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.42it/s]
Epoch 18/20 - loss: 1.0319 - acc: 0.9911 - val_loss: 1.0995 - val_acc: 0.9589
Epoch 19 Train: 100%|█████████▉| 3445/3446 [11:31<00:00,  5.00it/s, loss=1.0275, acc=0.9924]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.47it/s]
Epoch 19/20 - loss: 1.0275 - acc: 0.9924 - val_loss: 1.0959 - val_acc: 0.9572
Epoch 20 Train: 100%|█████████▉| 3445/3446 [11:31<00:00,  4.99it/s, loss=1.0252, acc=0.9931]
Validate: 100%|██████████| 607/607 [00:40<00:00, 15.47it/s]
Epoch 20/20 - loss: 1.0252 - acc: 0.9931 - val_loss: 1.0901 - val_acc: 0.9601
Completed successfully.
Reloading model and classes to apply new weights...
Model and classes reloaded successfully.
