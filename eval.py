from rouge import Rouge
from transformers import T5Tokenizer
from tqdm import tqdm
from model.parse_data import parse_data_improved
import pandas as pd

from train import TEXT_LEN, SUM_LEN, model_name, MAX_SENTENCE_PER_SEC
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
    TOKENIZER = T5Tokenizer.from_pretrained(model_name)
    resume_ckpt = 'lightning_logs/version_1254483/checkpoints/epoch=6-step=1120.ckpt'
    summary_model = SummaryModel.load_from_checkpoint(resume_ckpt)
    summary_model.eval()

    # prepare test data
    fd_test = open('data/test.txt', 'r')
    test_list = fd_test.read().split()
    fd_test.close()

    test_data = []
    # parse_data_base(test_list, test_data)
    parse_data_improved(test_list, test_data, MAX_SENTENCE_PER_SEC)
    test_df = pd.DataFrame(test_data, columns=['Summary', 'Text'])

    gts = []
    pred = []
    f_gt = open('sum_gt.txt', 'w')
    f_pred = open('sum_pred.txt', 'w')
    for i in tqdm(range(len(test_df['Text'])), desc='Infer'):
        text = str(test_df['Text'][i])
        gt_sum = str(test_df['Summary'][i])
        gts.append(gt_sum)
        pred_sum = summarize(text)
        pred.append(pred_sum)

        f_gt.write(f'{i+1} GT: {gt_sum}\n\n\n')
        f_pred.write(f'{i+1} PRED: {pred_sum}\n\n\n')
        print(f'GT: [{gt_sum}]')
        print(f'PRED: [{pred_sum}]')

    f_gt.close()
    f_pred.close()

    scores = rouge_score.get_scores(pred, gts, avg=True)
    print(scores)



