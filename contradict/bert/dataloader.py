from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", do_lower_case=True)
MAX_LEN = 512

class CompDataset(Dataset):
    def __init__(self, df):
        self.df_data = df

    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence1 = self.df_data.loc[index, 'premise']
        sentence2 = self.df_data.loc[index, 'hypothesis']
        # Process the sentence
        # ---------------------
        encoded_dict = tokenizer.encode_plus(
                    sentence1, sentence2,           # Sentences to encode.
                    add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                    max_length = MAX_LEN,           # Pad or truncate all sentences.
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',          # Return pytorch tensors.
               )  
        # These are torch tensors already.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]
        # Convert the target to a torch tensor
        target = torch.tensor(self.df_data.loc[index, 'label']).to("cuda")
        sample = (padded_token_list, att_mask, token_type_ids, target)
        return sample


    def __len__(self):
        return len(self.df_data)


class TestDataset(Dataset):
    def __init__(self, df):
        self.df_data = df

    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence1 = self.df_data.loc[index, 'premise']
        sentence2 = self.df_data.loc[index, 'hypothesis']
        # Process the sentence
        # ---------------------
        encoded_dict = tokenizer.encode_plus(
                    sentence1, sentence2,           # Sentence to encode.
                    add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                    max_length = MAX_LEN,           # Pad or truncate all sentences.
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',          # Return pytorch tensors.
               )
        # These are torch tensors already.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]
        sample = (padded_token_list, att_mask, token_type_ids)
        return sample

    def __len__(self):
        return len(self.df_data)