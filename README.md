# Prodigy_AI_T01
Here's a step-by-step guide on fine-tuning GPT-2 using Gradio in Python:

Install Required Libraries


bash
pip install transformers gradio torch


Import Libraries


import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


Load Pre-trained GPT-2 Model and Tokenizer


model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


Prepare Custom Dataset


# Load your custom dataset (e.g., a list of strings)
train_data = [...]  # replace with your dataset

# Preprocess the data (e.g., tokenize and pad)
inputs = tokenizer(train_data, padding=True, truncation=True, return_tensors='pt')


Fine-Tune GPT-2 Model


# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(link unavailable)(device)

# Define fine-tuning hyperparameters
batch_size = 16
epochs = 3
learning_rate = 1e-5

# Create a custom dataset class for our data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.inputs.items()}

# Create data loaders
dataset = CustomDataset(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fine-tune the model
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['input_ids'].to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()


Create a Gradio Interface


def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Textbox(label="Generated Text"),
    title="GPT-2 Text Generator",
    description="Enter a prompt to generate text",
)

iface.launch()
