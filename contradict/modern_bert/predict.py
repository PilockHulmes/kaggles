import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from .dataloader import CompDataset, TestDataset
import gc

device = "cuda"

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=3,
    )
    model.load_state_dict(torch.load('mordern_bert_contradict.pt'))
    model.to(device)
    model.eval()

    df_test = pd.read_csv("./contradict/data/test.csv")
    test_dataloader = DataLoader(TestDataset(df_test), batch_size=8, shuffle=True)

    total_predictions = []
    for j, batch in enumerate(test_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)  # 形状 [batch_size]
        total_predictions = total_predictions + predicted_labels.tolist()
    output = pd.DataFrame({
        "id": df_test["id"],
        "prediction": total_predictions,
    })
    output.to_csv("./contradict/submission.csv", index=False)