import os
import pickle
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load item types from item_to_pc.csv
pc_to_item = {}
with open("item_to_pc.csv", "r") as file:
    lines = file.readlines()
    pc_to_item = {int(line.strip().split(",")[1]): line.strip().split(",")[0] for line in lines if line.strip()}

# Load aliases from alias_to_pc.csv
alias_to_pc = {}
with open("alias_to_pc.csv", "r") as file:
    lines = file.readlines()
    alias_to_pc = {line.strip().split(",")[0]: int(line.strip().split(",")[1]) for line in lines if line.strip()}
    aliases = list(alias_to_pc.keys())


# Calculate embeddings
embeddings = model.encode(aliases)


# Save the data to pickle files
# with open("sentence_transformer_model.pkl", "wb") as f:
#     pickle.dump(model, f)
    
with open("pc_to_item.pkl", "wb") as f:
    pickle.dump(pc_to_item, f)
    
with open("alias_to_pc.pkl", "wb") as f:
    pickle.dump(alias_to_pc, f)
    
with open("aliases.pkl", "wb") as f:
    pickle.dump(aliases, f)
    
with open("alias_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
