# Information About ScisummNet

The dataset contains 1009 papers with summaries.  The size of the training, validation and test set is 801, 99 and 109 respectively.

The paper content is in *data/top1000_complele/paper_id/Documents_xml/paper_id.xml*, written not in plain text. Maybe we need to 
preprocess the paper content? But I believe at least for deep learning based model, it should be also fine if we don't preprocess the content.

The summary is in *data/top1000_complele/paper_id/summary/paper_id.gold.txt*. We can directly use it as supervision to train the model.
They are manually selected and annotated.

The *data/top1000_complele/paper_id/citing_sentences_annotated.json* file stores many one-sentence summaries from other papers that cite the current paper.
Some are them are of good quality.

In the paper of ScisummNet, they take the abstract of the paper from .xml file, together with all the cleaned citing sentences as input, and use the 
annotated summary as supervision.
