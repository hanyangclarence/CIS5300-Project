# Running the Evaluation Script
To evaluate the performance of the summarization model using the ROUGE metric, follow the steps below.

# Prerequisites
Ensure you have the rouge Python package installed. If not, you can install it via pip:

`pip install rouge`

# Script Usage
## Prepare your data: 
Ensure you have two text files - one containing the system-generated summaries and the other containing the reference summaries.

## Run the Script: 
Use the command line to run the evaluation script. Replace system_output.txt with the path to your system's output file and gold_standard.txt with the path to your gold standard file.

`python score.py --system_output system_output.txt --gold_standard gold_standard.txt`

# Example Command
Assuming you have system_output.txt and gold_standard.txt in your current directory, run the script as follows:

`python score.py --system_output system_output.txt --gold_standard gold_standard.txt`

# Example Output
The script outputs the average ROUGE scores. An example output might look like this:

`{
    "rouge-1": {
        "f": 0.47863,
        "p": 0.48684,
        "r": 0.47059
    },
    "rouge-2": {
        "f": 0.26068,
        "p": 0.26666,
        "r": 0.25490
    },
    "rouge-l": {
        "f": 0.47863,
        "p": 0.48684,
        "r": 0.47059
    }
}`

This output shows the F1-score (f), precision (p), and recall (r) for ROUGE-1, ROUGE-2, and ROUGE-L.
