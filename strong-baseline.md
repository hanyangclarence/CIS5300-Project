# Running the Strong-Baseline Script

This is a strong baseline that we implemented with the base model of the pretrained T5 model. 

# Prerequisites

Please make sure that you have all the packages, listed on the top of the strong_baseline.py file, installed in your environment. 

# Script Usage
## Prepare your data: 
The first part of the code will preprocess the data into a form that can be interpreted by the model. 

## Run the Script: 
### Parameters to adjust:
BATCH_SIZE = 4\
TEXT_LEN = 1024\
SUM_LEN = 256\
EPOCHS = 2

## Output
The file will output an average Rouge-L score for all the test data. 