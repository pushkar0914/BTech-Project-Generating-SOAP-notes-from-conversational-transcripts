## /models/noteworthy_extractor.py
import torch
import torch.nn as nn
from transformers import BertModel

class NoteworthyExtractor(nn.Module):
    def __init__(self):
        super(NoteworthyExtractor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 15)  # Output matches 16 SOAP subsections

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, (hidden_state, _) = self.lstm(bert_output)
        
        # Concatenating last hidden states from both directions
        hidden_cat = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        output = self.fc(hidden_cat)  # Convert to 15-section predictions
        return output
