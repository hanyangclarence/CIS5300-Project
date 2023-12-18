import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

fd_val = open('data/val.txt', 'r')
val_list = fd_val.read().split()
fd_val.close()

val_data = []
for i in tqdm(range(len(val_list)), desc='Val data'):
    file = val_list[i]
    try:
        tree = ET.parse("data/top1000_complete/" + file + "/Documents_xml/" + file + ".xml")
        root = tree.getroot()
        xmlstr = " ".join([elem.text for elem in root.findall('.//S') if elem.text is not None])

        summary = open('data/top1000_complete/' + file + "/summary/" + file + ".gold.txt", 'r')
        summary_str = summary.read()
        summary.close()

        val_data.append([summary_str, xmlstr])
    except:
        pass

val_df = pd.DataFrame(val_data, columns=['Summary', 'Text'])