# Running the Extension 1 Script

# Prerequisites

We finetune the model using pytorch-lightning pipeline, so please install the corresponding packages listed in `train.py`, 
`eval.py` and `model` folder. We train the model on a 32G V100 GPU, so you can modify the batch size and text length if 
there is not enough GPU memories on your device.

# Script Usage
## Data: 
All the data is contained in `data` folder. There is no need to further download anything.

## Train the model:
Run `python train.py` on terminal.
### Parameters to adjust:
model_name = 't5-base' \
TEXT_LEN = 1024 \
SUM_LEN = 512 \
BATCH_SIZE = 5 \
EPOCHS = 10 

## Evaluate the model
The best checkpoint is automatically saved by pytorch-lightning in folder 
`lightning_logs/version_0/checkpoints/epoch=xxx-step=xxx.ckpt`. First set this as the `resume_ckpt` 
in line 32 of `eval.py`, then run `python eval.py` on terminal. The script will write the predicted summary
and ground truth summary into `sum_pred.txt` and `sum_gt.txt` respectively, and print the averaged Rouge scores. 