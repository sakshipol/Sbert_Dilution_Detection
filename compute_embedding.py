import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("buzzword_dilution_dataset.csv",encoding="latin1")

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

# Save embeddings
np.save("text_embeddings.npy", embeddings)

print("Embeddings saved successfully!")
