import torch.cuda
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from rouge import Rouge
from model.model import SummaryModelLongT5
from model.dataset import SummaryDataModule
from model.parse_data import parse_data_longT5

# Hyperparameters
model_name = "Stancld/longt5-tglobal-base-16384-pubmed-3k_steps"
TEXT_LEN = 2048
SUM_LEN = 512
BATCH_SIZE = 5
EPOCHS = 10
MAX_SENTENCE_PER_SEC = 2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def summarize(text_input):
    inputs = TOKENIZER(text_input,
                       max_length=TEXT_LEN,
                       truncation=True,
                       padding="max_length",
                       add_special_tokens=True,
                       return_tensors="pt")
    summarized_ids = summary_model.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=1,
        do_sample=False,
        max_length=SUM_LEN)

    return " ".join([TOKENIZER.decode(token_ids, skip_special_tokens=True)
                     for token_ids in summarized_ids])


if __name__ == "__main__":
    seed_everything(23)

    # Preprocessing
    fd_train = open('data/train.txt', 'r')
    fd_test = open('data/test.txt', 'r')
    fd_val = open('data/val.txt', 'r')

    train_list = fd_train.read().split()
    test_list = fd_test.read().split()
    val_list = fd_val.read().split()

    fd_train.close()
    fd_test.close()
    fd_val.close()

    train_data = []
    # parse_data_base(train_list, train_data)
    parse_data_longT5(train_list, train_data, MAX_SENTENCE_PER_SEC)
    train_df = pd.DataFrame(train_data, columns=['Summary', 'Text'])

    test_data = []
    # parse_data_base(test_list, test_data)
    parse_data_longT5(test_list, test_data, MAX_SENTENCE_PER_SEC)
    test_df = pd.DataFrame(test_data, columns=['Summary', 'Text'])

    val_data = []
    # parse_data_base(val_list, val_data)
    parse_data_longT5(val_list, val_data, MAX_SENTENCE_PER_SEC)
    val_df = pd.DataFrame(val_data, columns=['Summary', 'Text'])

    print(f'Train: {len(train_data)}, Test: {len(test_data)}, Val: {len(val_data)}')

    # Final dataframe
    df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # setup dataset module
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    summary_module = SummaryDataModule(train_df, val_df, test_df, BATCH_SIZE, TOKENIZER, TEXT_LEN, SUM_LEN)
    summary_module.setup()
    next(iter(summary_module.train_dataloader().dataset))

    # setup model
    summary_model = SummaryModelLongT5(model_name=model_name, total_step=EPOCHS * len(df))

    # setup checkpoint saver
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1)

    # setup trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        callbacks=[checkpoint_callback]
    )

    trainer.fit(summary_model, summary_module)

    print(f'Train done, best model saved in {checkpoint_callback.best_model_path}')

    # evaluate the best model
    rouge_score = Rouge()
    summary_model = SummaryModelLongT5.load_from_checkpoint(checkpoint_callback.best_model_path)
    summary_model.eval()

    gts = []
    pred = []
    f_gt = open('sum_gt.txt', 'w')
    f_pred = open('sum_pred.txt', 'w')
    for i in tqdm(range(len(test_df['Text'])), desc='Infer'):
        text = test_df['Text'][i]
        gt_sum = test_df['Summary'][i]
        gts.append(gt_sum)
        pred_sum = summarize(text)
        pred.append(pred_sum)

        f_gt.writelines(gt_sum)
        f_pred.writelines(pred_sum)
        print(f'GT: [{gt_sum}]')
        print(f'PRED: [{pred_sum}]')

    f_gt.close()
    f_pred.close()

    scores = rouge_score.get_scores(pred, gts, avg=True)
    print(scores)












