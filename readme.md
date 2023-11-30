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

The content of the papers is located in the directory *data/top1000_complele/paper_id/Documents_xml/paper_id.xml*. 
It is important to note that the format of these documents is not plain text. Considering this, we probably need to preprocess the paper content to facilitate more effective data handling. However, for models based on deep learning techniques, it could be feasible to proceed without preprocessing.
- Example document:
```
<PAPER>
  <S sid="0">TnT - A Statistical Part-Of-Speech Tagger</S>
  <ABSTRACT>
    <S sid="1" ssid="1">Trigrams'n'Tags (TnT) is an efficient statistical part-of-speech tagger.</S>
    ...
  </ABSTRACT>
  <SECTION title="1 Introduction" number="1">
    <S sid="6" ssid="1">[Introduction content...]</S>
    ...
  </SECTION>
  ...
</PAPER>
```

The summaries are located in *data/top1000_complele/paper_id/summary/paper_id.gold.txt*. 
We can directly use it as supervision for training the model. They are manually selected and annotated.
- Example gold summary:
```
TnT - A Statistical Part-Of-Speech Tagger
Trigrams'n'Tags (TnT) is an efficient statistical part-of-speech tagger.
Contrary to claims found elsewhere in the literature, we argue that a tagger based on Markov models performs at least as well as other current approaches, including the Maximum Entropy framework.
A recent comparison has even shown that TnT performs significantly better for the tested corpora.
We describe the basic model of TnT, the techniques used for smoothing and for handling unknown words.
Furthermore, we present evaluations on two corpora.
We achieve the automated tagging of a syntactic-structure-based set of grammatical function tags including phrase-chunk and syntactic-role modifiers trained in supervised mode from a tree bank of German.
```

The file located at *data/top1000_complele/paper_id/citing_sentences_annotated.json* contains a collection of one-sentence summaries extracted from other papers that cite the current paper. Some are them are of good quality in advancing our model.
- Example citing sentence:
```
[
  {
    "citance_No": 1, 
    "citing_paper_id": "W00-1326", 
    "citing_paper_authority": 13, 
    "citing_paper_authors": "David, Martinez | Eneko, Agirre", 
    "raw_text": "The sentences in the DSO collection were tagged with parts of speech using TnT (Brants, 2000) trained on the Brown Corpus itself", 
    "clean_text": "The sentences in the DSO collection were tagged with parts of speech using TnT (Brants, 2000) trained on the Brown Corpus itself.", 
    "keep_for_gold": 0
  },
  ...
]
```

In the paper of ScisummNet, they take the abstract of the paper from .xml file, together with all the cleaned citing sentences as input, and use the 
annotated summary as supervision.
