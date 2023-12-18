import torch.cuda
from tqdm import tqdm
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import xml.etree.ElementTree as ET
import pytorch_lightning as pl

from model.model import SummaryModel
from model.dataset import SummaryDataModule

if __name__ == "__main__":

    # Hyperparameters
    MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    TOKENIZER = T5Tokenizer.from_pretrained("t5-base")
    BATCH_SIZE = 5
    TEXT_LEN = 1024
    SUM_LEN = 512
    EPOCHS = 10
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    for i in tqdm(range(len(train_list)), desc="Training data"):
        file = train_list[i]
        try:
            tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
            root = tree.getroot()
            xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

            summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
            summary_str = summary.read()
            summary.close()

            train_data.append([summary_str, xmlstr])
        except:
            pass

    train_df = pd.DataFrame(train_data, columns=['Summary', 'Text'])

    test_data = []
    for i in tqdm(range(len(test_list)), desc='Test data'):
        file = test_list[i]
        try:
            tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
            root = tree.getroot()
            xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

            summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
            summary_str = summary.read()
            summary.close()

            test_data.append([summary_str, xmlstr])
        except:
            pass

    test_df = pd.DataFrame(test_data, columns=['Summary', 'Text'])

    val_data = []
    for i in tqdm(range(len(val_list)), desc='Val data'):
        file = val_list[i]
        try:
            tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
            root = tree.getroot()
            xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

            summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
            summary_str = summary.read()
            summary.close()

            val_data.append([summary_str, xmlstr])
        except:
            pass

    val_df = pd.DataFrame(val_data, columns=['Summary', 'Text'])

    # Final dataframe
    df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # setup dataset module
    summary_module = SummaryDataModule(train_df, val_df, test_df, BATCH_SIZE, TOKENIZER, TEXT_LEN, SUM_LEN)
    summary_module.setup()
    next(iter(summary_module.train_dataloader().dataset))

    # setup model
    summary_model = SummaryModel(model=MODEL, total_step=EPOCHS * len(df))

    # setup trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
    )

    trainer.fit(summary_model, summary_module)
















