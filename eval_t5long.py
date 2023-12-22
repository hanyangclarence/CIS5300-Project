from rouge import Rouge
from transformers import AutoTokenizer
from tqdm import tqdm
from model.parse_data import parse_data_longT5
import pandas as pd

from train_t5long import TEXT_LEN, SUM_LEN, model_name, MAX_SENTENCE_PER_SEC
from model.model import SummaryModelLongT5


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
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    resume_ckpt = 'lightning_logs/version_1254484/checkpoints/epoch=12-step=5187.ckpt'
    summary_model = SummaryModelLongT5.load_from_checkpoint(resume_ckpt)
    summary_model.eval()

    # prepare test data
    fd_test = open('data/test.txt', 'r')
    test_list = fd_test.read().split()
    fd_test.close()

    test_data = []
    # parse_data_base(test_list, test_data)
    parse_data_longT5(test_list, test_data, MAX_SENTENCE_PER_SEC)
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

        f_gt.write(gt_sum + '\n\n\n')
        f_pred.write(pred_sum + '\n\n\n')
        print(f'GT: [{gt_sum}]')
        print(f'PRED: [{pred_sum}]')

        scores = rouge_score.get_scores(pred, gts, avg=True)
        print(scores)

    f_gt.close()
    f_pred.close()

    scores = rouge_score.get_scores(pred, gts, avg=True)
    print(scores)



