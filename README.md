# 🧠 Mental Health Assistant with Llama-3.1 & LoRA

This repository contains a specialized AI assistant designed to provide motivational and actionable support for mental health concerns. 
The project utilizes Parameter-Efficient Fine-Tuning (PEFT) on the Llama-3.1-8B-Instruct model to deliver concise and encouraging responses.

---

## 🌟 Key Features

* **Model Architecture:** Built on top of Meta's Llama-3.1-8B-Instruct, a state-of-the-art large language model.
* **Fine-Tuning Technique:** Employs LoRA (Low-Rank Adaptation) for efficient model adaptation, significantly reducing the memory footprint while maintaining performance.
* **Dataset Integration:** Combines multiple data sources (JSON and CSV) related to mental health intents and QA pairs for a comprehensive knowledge base.
* **Structured Prompting:** Uses a specialized system prompt to ensure responses are concise, actionable, and always conclude with a motivational message.
* **Interactive UI:** Features a built-in Gradio web interface for real-time interaction, allowing users to input age, gender, profession, and their specific issue.

## 🛠 Technical Stack

| Category | Tools |
| :--- | :--- |
| **Framework** | Hugging Face Transformers, PEFT, Datasets |
| **Model** | Llama-3.1-8B-Instruct |
| **Deployment/UI** | Gradio  |
| **Environment** | Google Colab / Python |

## 📂 Data Source

The model's knowledge base is built by merging three specialized sources: intents.json provides the foundation for natural conversational flow and basic intent recognition; Mental_Health_QA.json contributes domain-specific expertise through expert-level question-and-answer pairs; and train.csv offers structured contextual dialogues that enable the assistant to provide personalized advice based on user demographics like age and profession. By normalizing these diverse JSON and CSV formats into a unified "Instruction-Context-Response" template, the system creates a comprehensive training dataset that balances social intelligence with clinical information and personal relevance.

## ⚠️ Disclaimer
**This AI assistant is for educational and motivational purposes only.**
It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.

---
