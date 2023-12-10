# Simple baseline model
This code is designed to evaluate the performance of a simple baseline summarization algorithm using the ROUGE-L metric

## Parameters to adjust:
- chunk_size=1000
- num_sentences=3

## Execution
The script sets paths for the document directory and the list of document IDs, then calls evaluate to compute the average ROUGE-L score of the summarization algorithm over the document set. Finally, it prints the average ROUGE-L score.
