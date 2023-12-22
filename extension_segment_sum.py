import torch
import torch.cuda
from tqdm import tqdm
import pandas as pd
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import os
from rouge import Rouge

class SummaryModel(pl.LightningModule):
    def __init__(self, model_name="t5-base", total_step=None):
        super(SummaryModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        self.total_step = total_step

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
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["summary_ids"]
        decoder_attention_mask = batch["summary_mask"]

        loss, output = self(input_ids, attention_mask, labels, decoder_attention_mask)
        self.log("val_loss", loss.item())

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss, output = self(input_ids=input_ids, attention_mask=attention_mask)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=self.total_step)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

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

def has_numeric(s):
    for c in s:
        if c.isnumeric():
            return True
    return False

def parse_data_base(source_data_list: list, parsed_res_list: list):
    for i in tqdm(range(len(source_data_list)), desc='Parsing'):
        file = source_data_list[i]
        try:
            tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
            root = tree.getroot()
            xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

            summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
            summary_str = summary.read()
            summary.close()

            parsed_res_list.append([summary_str, xmlstr])
        except:
            pass

def parse_data_improved(source_data_list: list, parsed_res_list: list):
    for i in tqdm(range(len(source_data_list)), desc='Parsing data'):
        file = source_data_list[i]
        xml_path = os.path.join('data/top1000_complete', file, 'Documents_xml', file + '.xml')
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()

        text = []
        # find the title
        title_element = root.find(".//S[@sid='0']")
        if title_element is not None and title_element.text is not None:
            text.append(title_element.text.strip() + '\n')

        sections = {}

        # traverse each section
        for child in root:
            if child.tag == 'ABSTRACT':
                # add all sentences in abstract
                for sub_child in child:
                    text.append(sub_child.text + '\n')
            elif child.tag == 'SECTION':
                section_title = child.attrib['title'].lower()
                if 'intro' in section_title:
                    # add all sentences in intro section
                    for sub_child in child:
                        text.append(sub_child.text + '\n')
                elif 'conclusion' in section_title:
                    # add all sentences in conclusion section
                    for sub_child in child:
                        text.append(sub_child.text + '\n')
                elif 'acknowledgement' in section_title:
                    # do not include anything in acknowledgement
                    continue
                elif section_title == '' and child.attrib['number'] == '1':
                    # on observation, these paragraphs are also important, just include all
                    for sub_child in child:
                        text.append(sub_child.text + '\n')
                else:
                    section_text = ' '.join([sub_child.text for sub_child in child if sub_child.text is not None])
                    sections[section_title] = section_text


        # filter out noisy sentences
        selected_text = []
        for i_sent, sent in enumerate(text):
            n_word = len(sent.split(' '))
            if n_word > 50:  # too many numbers
                if sum([has_numeric(w) for w in sent.split(' ')]) / n_word > 0.15:
                    continue
            if n_word <= 4:  # too short
                if i_sent != 0:
                    continue
            if n_word > 30:  # too many symbols/incomplete words
                if sum([len(w) <= 2 for w in sent.split(' ')]) / n_word > 0.4:
                    continue
            if n_word > 20:  # contains too many '/'
                if sum([('/' in w) for w in sent.split(' ')]) / n_word > 0.2:
                    continue
            if sent in selected_text:  # remove duplicated sentences
                continue
            selected_text.append(sent)

        final_text = ''.join(selected_text)

        summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'rb')
        summary_str = summary.read()
        summary_str = summary_str.decode('utf-8')
        summary.close()

        parsed_res_list.append([summary_str, final_text, sections])

# Hyperparameters
model_name = 't5-base'
TEXT_LEN = 1024
SUM_LEN = 512
BATCH_SIZE = 3
EPOCHS = 10
MAX_SENTENCE_PER_SEC = 2
AVG_GT_WORDS = 150
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_section_word_limits(sections, total_word_limit=150):
    num_sections = len(sections)
    if num_sections == 0: return {}

    limit_per_section = total_word_limit // num_sections
    return {title: limit_per_section for title in sections}

def summarize_segment(model, tokenizer, text_input, max_length, max_summary_length):
    inputs = tokenizer("summarize: " + text_input,
                       max_length=max_length,
                       truncation=True,
                       padding="max_length",
                       add_special_tokens=True,
                       return_tensors="pt")
    with torch.no_grad():
        summarized_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=1,
            do_sample=False,
            max_length=max_summary_length)
    return tokenizer.decode(summarized_ids[0], skip_special_tokens=True)

def summarize_document(model, tokenizer, text, sections, text_len, sum_len, AVG_GT_WORDS):
    section_word_limits = compute_section_word_limits(sections, AVG_GT_WORDS - len(text.split()))

    summarized_sections = []
    for title, text in sections.items():
        words_limit = section_word_limits.get(title, sum_len)
        summarized_section = summarize_segment(model, tokenizer, text, text_len, words_limit)
        summarized_sections.append(summarized_section)

    final_summary = ' '.join([text] + summarized_sections)
    final_summary_words = final_summary.split()
    # return summarize_segment(model, tokenizer, final_summary, text_len, len(text.split()) + AVG_GT_WORDS)
    return ' '.join(final_summary_words[:AVG_GT_WORDS])

if __name__ == "__main__":
    seed_everything(23)

    # Preprocessing
    fd_test = open('data/test.txt', 'r')
    test_list = fd_test.read().split()
    fd_test.close()

    test_data = []
    parse_data_improved(test_list, test_data)
    test_df = pd.DataFrame(test_data, columns=['Summary', 'Text', 'Sections'])

    print(f'Test: {len(test_data)}')

    # Final dataframe
    test_df = test_df.reset_index(drop=True)

    # setup dataset module
    TOKENIZER = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # evaluate the base model
    rouge_score = Rouge()

    gts = []
    pred = []
    f_gt = open('sum_gt.txt', 'w')
    f_pred = open('sum_pred.txt', 'w')
    for i in tqdm(range(len(test_df)), desc='Infer'):
        text, sections = test_df['Text'][i], test_df['Sections'][i]
        gt_sum = test_df['Summary'][i]

        pred_sum = summarize_document(model, TOKENIZER, text, sections, TEXT_LEN, SUM_LEN, AVG_GT_WORDS)

        if len(pred_sum) == 0:
            continue
        pred.append(pred_sum)
        gts.append(gt_sum)

        f_gt.writelines(gt_sum)
        f_pred.writelines(pred_sum)
        print(f'GT: [{gt_sum}]')
        print(f'PRED: [{pred_sum}]')

    f_gt.close()
    f_pred.close()

    scores = rouge_score.get_scores(pred, gts, avg=True)
    print(scores)