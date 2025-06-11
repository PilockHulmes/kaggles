import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from .dataloader import CompDataset, TestDataset
import gc

L_RATE = 1e-5
MAX_LEN = 256
NUM_EPOCHS = 3
BATCH_SIZE = 32

device = "cuda"
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=3,
)

model.to(device)
df_train = pd.read_csv("./contradict/data/train.csv")
df_test = pd.read_csv("./contradict/data/train.csv")

train_dataloader = torch.utils.data.DataLoader(df_train, batch_size=8, shuffle=True, num_workers=1)
test_dataloader = torch.utils.data.DataLoader(df_test, batch_size=8, shuffle=True, num_workers=1)

# batch = next(iter(train_dataloader))

# b_input_ids = batch[0].to(device)
# b_input_mask = batch[1].to(device)
# b_token_type_ids = batch[2].to(device)
# b_labels = batch[3].to(device)

# outputs = model(b_input_ids, 
#                 token_type_ids=b_token_type_ids, 
#                 attention_mask=b_input_mask,
#                 labels=b_labels)

optimizer = AdamW(model.parameters(),
              lr = L_RATE, 
              eps = 1e-8 
            )

for epoch in range(NUM_EPOCHS):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))
    print('Training...')

    # put the model into train mode
    model.train()
    # This turns gradient calculations on and off.
    torch.set_grad_enabled(True)
    # Reset the total loss for this epoch.
    total_train_loss = 0

    for i, batch in enumerate(train_dataloader):
        train_status = 'Batch ' + str(i) + ' of ' + str(len(train_dataloader))
        print(train_status, end='\r')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        # Get the loss from the outputs tuple: (loss, logits)
        loss = outputs[0]
        # Convert the loss from a torch tensor to a number.
        # Calculate the total loss.
        total_train_loss = total_train_loss + loss.item()
        # Zero the gradients
        optimizer.zero_grad()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Optimizer for GPU
        optimizer.step()

    print('Train loss:' ,total_train_loss)
    # Save the Model
    torch.save(model.state_dict(), 'mordern_bert_contradict.pt')
    # Use the garbage collector to save memory.
    gc.collect()

targets_list = []
for j, batch in enumerate(test_dataloader):
        
        inference_status = 'Batch ' + str(j+1) + ' of ' + str(len(test_dataloader))
        print(inference_status, end='\r')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        outputs = model(b_input_ids, attention_mask=b_input_mask)

        # Get the preds
        preds = outputs[0]

        # Move preds to the CPU
        preds = preds.detach().cpu().numpy()
        
        # Move the labels to the cpu
        targets_np = b_labels.to('cpu').numpy()

        # Append the labels to a numpy list
        targets_list.extend(targets_np)
        
        # Stack the predictions.

        if j == 0:  # first batch
            stacked_preds = preds

        else:
            stacked_preds = np.vstack((stacked_preds, preds))