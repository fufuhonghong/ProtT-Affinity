import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

# ===============================


os.environ['HF_HOME'] = r"D:\huggingface_cache"  #  
print(f"🔹 Hugging Face : {os.environ['HF_HOME']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" : {device}")

# ===============================


LOCAL_MODEL_PATH = os.path.join(os.environ['HF_HOME'], "hub", "models--Rostlab--prot_t5_xl_uniref50")
if os.path.exists(LOCAL_MODEL_PATH):
    model_name_or_path = LOCAL_MODEL_PATH
   
else:
    model_name_or_path = "Rostlab/prot_t5_xl_uniref50"
    print(f"  Hugging Face : {model_name_or_path}")

# ===============================

tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_name_or_path).to(device)
model.eval()

# ===============================


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    seq = str(seq).upper()
    return "".join([aa if aa in STANDARD_AA else "X" for aa in seq])

def get_prott5_embedding(seq):
    seq = clean_sequence(seq)
    seq = " ".join(list(seq))  
    ids = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    ids = ids.to(device)
    with torch.no_grad():
        emb = model(**ids).last_hidden_state.mean(dim=1)
    return emb.squeeze().cpu().numpy() 

# ===============================

df = pd.read_csv("./test2.csv")  

embA_list, embB_list = [], []
for i, row in tqdm(df.iterrows(), total=len(df)):
    pdbid = str(row['pdb_id']).upper()
    try:
        embA = get_prott5_embedding(row['seqA'])
        embB = get_prott5_embedding(row['seqB'])
        embA_list.append(embA)
        embB_list.append(embB)
    except Exception as e:
        embA_list.append(None)
        embB_list.append(None)

df['embA'] = embA_list
df['embB'] = embB_list
df = df.dropna(subset=['embA', 'embB']).reset_index(drop=True)

output_file = "./test2_embeddings.pkl"
df.to_pickle(output_file)

