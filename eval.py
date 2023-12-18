from rouge import Rouge
from transformers import T5Tokenizer
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pandas as pd

from train import TEXT_LEN, SUM_LEN
from model.model import SummaryModel


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
        num_beams=1,
        do_sample=False,
        max_length=SUM_LEN)

    return " ".join([TOKENIZER.decode(token_ids, skip_special_tokens=True)
                     for token_ids in summarized_ids])


if __name__ == "__main__":
    rouge_score = Rouge()
    TOKENIZER = T5Tokenizer.from_pretrained("t5-base")
    resume_ckpt = ''
    summary_model = SummaryModel.load_from_checkpoint(resume_ckpt)

    # prepare test data
    fd_test = open('data/test.txt', 'r')
    test_list = fd_test.read().split()
    fd_test.close()

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



