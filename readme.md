# Academic Paper Summarization Project

## Dataset

We use the [Yale Scientific Article Summarization Dataset (ScisummNet)](https://cs.stanford.edu/~myasu/projects/scisumm_net/) for our project, which incorporates 1,009 papers in the ACL anthology network with their citation networks (e.g. citation sentences, citation counts) and their comprehensive, manual summaries (gold summaries). 

We split the dataset into three parts:
- training set: 801 entries
- validation set: 99 entries
- test set: 109 entries

For each reference reference paper:
- average length of total text: 4417 words 
- average length of gold summary: 151 words 

The content of the papers is located in the directory *data/top1000_complele/paper_id/Documents_xml/paper_id.xml*. It is important to note that the format of these documents is not plain text. Considering this, we probably need to 
preprocess the paper content to facilitate more effective data handling. However, for models based on deep learning techniques, it could be feasible to proceed without preprocessing.

The summaries are located in *data/top1000_complele/paper_id/summary/paper_id.gold.txt*. We can directly use it as supervision for training the model. They are manually selected and annotated.

The file located at *data/top1000_complele/paper_id/citing_sentences_annotated.json* contains a collection of one-sentence summaries extracted from other papers that cite the current paper. 
Some are them are of good quality.

In the paper of ScisummNet, they take the abstract of the paper from .xml file, together with all the cleaned citing sentences as input, and use the 
annotated summary as supervision.
