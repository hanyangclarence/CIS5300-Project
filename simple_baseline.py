import os
import re
import xml.etree.ElementTree as ET
from rouge import Rouge

def read_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])
        return text
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return ""

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def simple_baseline_summarizer(text, num_sentences=3):
    sentences = re.split(r'\.\s', text)
    return ' '.join(sentences[:num_sentences])

def rouge_l_score(baseline_summary, label_summary):
    rouge = Rouge()
    scores = rouge.get_scores(baseline_summary, label_summary)
    return scores[0]['rouge-l']['f']

def evaluate(doc_list, doc_path, num_sentences=3, chunk_size=1000):
    rouge_l_scores = []

    for paper_id in doc_list:
        xml_doc_path = os.path.join(doc_path, paper_id, 'Documents_xml', paper_id + '.xml')
        summary_label_path = os.path.join(doc_path, paper_id, 'summary', paper_id + '.gold.txt')

        doc_text = read_xml_file(xml_doc_path)
        ground_truth_summary = read_text_file(summary_label_path)

        if not doc_text.strip():  # Skip if doc is empty
            continue

        print(f"Processing document {paper_id}")

        chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size)]
        for chunk in chunks:
            extracted_summary = simple_baseline_summarizer(chunk, num_sentences)

            if not extracted_summary.strip():  # Skip if the summary is empty
                print(f"Empty summary for document {paper_id}")
                continue

            try:
                score = rouge_l_score(extracted_summary, ground_truth_summary)
                rouge_l_scores.append(score)
            except ValueError as e:
                print(f"Error calculating Rouge score for {paper_id}: {e}") # FIXME: P05-1074: Hypothesis is empty.

    return sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0

doc_path = 'data/top1000_complete'
doc_list = read_text_file('data/train.txt').splitlines()
avg_rouge_l_score = evaluate(doc_list, doc_path, num_sentences=3, chunk_size=1000)
print("Average Rouge-L Score:", avg_rouge_l_score)