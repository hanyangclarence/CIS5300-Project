import warnings
warnings.filterwarnings("ignore")
import torch
import re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lightning as pl
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download('omw-1.4')
import os
import string
import torch.nn as nn
import pandas as pd
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sentencepiece
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rouge import Rouge


import xml.etree.ElementTree as ET


##### Preprocessing
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
for file in train_list:
    tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
    root = tree.getroot()
    xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

    summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
    summary_str = summary.read()
    summary.close()

    train_data.append([summary_str, xmlstr])

train_df = pd.DataFrame(train_data, columns=['Summary', 'Text'])

test_data = []
for file in test_list:
    tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
    root = tree.getroot()
    xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

    summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
    summary_str = summary.read()
    summary.close()

    test_data.append([summary_str, xmlstr])

test_df = pd.DataFrame(test_data, columns=['Summary', 'Text'])

val_data = []
for file in val_list:
    tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
    root = tree.getroot()
    xmlstr = ET.tostring(root, encoding='utf8', method='xml')

    summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
    summary_str = summary.read()
    summary.close()

    val_data.append([summary_str, xmlstr])

val_df = pd.DataFrame(val_data, columns=['Summary', 'Text'])


### Final dataframe
df = pd.concat([train_df, test_df, val_df], ignore_index = True)


class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer, text_len, sum_len):
        self.df = df
        self.summaries = self.df["Summary"]
        self.text = self.df["Text"]
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.sum_len = sum_len
  
    def __len__(self):
        return len(self.summaries)
  
    def __getitem__(self, idx):
        # T5 transformers performs different tasks by prepending the particular prefix to the input text.
        text = "summarize:" + str(self.text[idx])                # In order to avoid dtype mismatch, as T5 is text-to-text transformer, the datatype must be string
        headline = str(self.summaries[idx])

        text_tokenizer = self.tokenizer(text, max_length=self.text_len, padding="max_length",
                                                        truncation=True, add_special_tokens=True)
        summary_tokenizer = self.tokenizer(headline, max_length=self.sum_len, padding="max_length",
                                                        truncation=True, add_special_tokens=True)
        return {
            "input_ids": torch.tensor(text_tokenizer["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(text_tokenizer["attention_mask"], dtype=torch.long),
            "summary_ids": torch.tensor(summary_tokenizer["input_ids"], dtype=torch.long),
            "summary_mask": torch.tensor(summary_tokenizer["attention_mask"], dtype=torch.long)
        }


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df,
                 val_df,
                 test_df,
                 batch_size,
                 tokenizer,
                 text_len,
                 summary_len):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.summary_len = summary_len

    def setup(self, stage=None):
        self.train_dataset = SummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_len,
            self.summary_len)

        self.val_dataset = SummaryDataset(
            self.val_df,
            self.tokenizer,
            self.text_len,
            self.summary_len)

        self.test_dataset = SummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_len,
            self.summary_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=8)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8
        )


class SummaryModel(pl.LightningModule):
    def __init__(self):
        super(SummaryModel, self).__init__()
        self.model = MODEL

    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             decoder_attention_mask=decoder_attention_mask)
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["summary_ids"]
        decoder_attention_mask = batch["summary_mask"]

        loss, output = self(input_ids, attention_mask, labels, decoder_attention_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["summary_ids"]
        decoder_attention_mask = batch["summary_mask"]

        loss, output = self(input_ids, attention_mask, labels, decoder_attention_mask)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss, output = self(input_ids=input_ids,
                            attention_mask=attention_mask)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.0001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=EPOCHS * len(df))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}




if __name__ == '__main__':
    MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    TOKENIZER = T5Tokenizer.from_pretrained("t5-base")
    BATCH_SIZE = 1
    TEXT_LEN = 64
    SUM_LEN = 32
    EPOCHS = 1
    DEVICE = "cuda:0"

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    summary_module = SummaryDataModule(train_df, val_df, test_df, BATCH_SIZE, TOKENIZER, TEXT_LEN, SUM_LEN)
    summary_module.setup()

    next(iter(summary_module.train_dataloader().dataset))

    model = SummaryModel()

    trainer = pl.Trainer(
        max_epochs=EPOCHS
    )

    trainer.fit(model, summary_module)


    # Loading the best model

    summary_model = SummaryModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    summary_model.freeze()

    def summarize(text):
        inputs = TOKENIZER(text,
                           max_length=TEXT_LEN,
                           truncation=True,
                           padding="max_length",
                           add_special_tokens=True,
                           return_tensors="pt")
        summarized_ids = summary_model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4)

        return " ".join([TOKENIZER.decode(token_ids, skip_special_tokens=True)
                        for token_ids in summarized_ids])




    ## Evaluation of the model

    hypothesis = []
    reference = []
    for lines in test_df:
        hypothesis.append(summarize(lines[1]))
        reference.append(lines[0])

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    print(scores)