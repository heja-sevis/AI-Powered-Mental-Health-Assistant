!pip install datasets transformers peft bitsandbytes accelerate gradio

import pandas as pd
from google.colab import drive
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')

data1 = pd.read_json('/content/drive/MyDrive/mentaldataset/intents.json')
data2 = pd.read_json('/content/drive/MyDrive/mentaldataset/Mental_Health_QA.json')
data3 = pd.read_csv('/content/drive/MyDrive/mentaldataset/train.csv')

data1 = data1.rename(columns={"intents": "instruction"})
data2 = data2.rename(columns={"intents": "instruction"})

data3 = data3.rename(columns={"Context": "context", "Response": "response"})

data1["context"] = ""
data1["response"] = ""

data2["context"] = ""
data2["response"] = ""


combined_data = pd.concat([data1, data2, data3], ignore_index=True)

print(combined_data.head())
from datasets import Dataset

dataset = Dataset.from_pandas(combined_data)

print(dataset)

def _add_text(rec):
    instruction = rec["instruction"]
    context = rec.get("context", "")
    response = rec.get("response", "")


    if context:
        rec["prompt"] = f"Instruction: {instruction}\nContext: {context}\nResponse:"
    else:
        rec["prompt"] = f"Instruction: {instruction}\nResponse:"

    rec["answer"] = response


    if rec["answer"] is None:
        rec["answer"] = ""

    rec["text"] = rec["prompt"] + rec["answer"]
    return rec

dataset = dataset.map(_add_text)
from huggingface_hub import login
login(token='')
from transformers import AutoTokenizer
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

from peft import get_peft_model, LoraConfig, TaskType
import torch
import gradio as gr

# LoRA Konfigürasyonu
define_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# LoRA modeli
lora_model = get_peft_model(model, define_config)

# Kısa ve yönlendirici bir sistem promptu
system_prompt = (
    "You are a helpful assistant. Be concise, actionable, and motivational. "
    "Always end the response with an encouraging and motivational message such as 'You got this!'. "
    "Avoid asking questions or prompting further user interactions. Conclude firmly and positively."
)

def generate_response(age, gender, profession, issue):

    if not age.strip() or not gender.strip() or not profession.strip() or not issue.strip():
        return "Please make sure you have filled in all fields: Age, Gender, Profession, Issue."


    user_input = f"Age: {age}\nGender: {gender}\nProfession: {profession}\nIssue: {issue}"
    full_input = f"{system_prompt}\nUser:\n{user_input}\nAssistant:"


    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = lora_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.85,
            top_k=30,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    response = response.split("Assistant:")[-1].strip()
    return response
  # Gradio arayüzü
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Age", placeholder="Example: 25,34 ..."),
        gr.Textbox(label="Gender", placeholder="Example: Male, Female ..."),
        gr.Textbox(label="Profession", placeholder="Example: Engineer, Student ..."),
        gr.Textbox(label="Issue", placeholder="Please specify the issue you are experiencing.")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Mental Health Assistant",
    description=(
        "Hello! I'm here to assist you. Could you please tell me about the issue you're currently facing? "
        "To help me understand better, it would also be great if you could share your gender, age, and profession."
    )
)

# Uygulamayı başlatma
demo.launch(server_name="0.0.0.0")
