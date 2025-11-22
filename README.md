
# 🧠 Custom GPT-2 Pretraining & Text Generation

This project fine-tunes **GPT-2** on custom domain datasets such as *Cricket*, *Education*, *Medical*, and more.
By training GPT-2 on your own `.txt` files, the model becomes better at generating domain-aware, context-rich text.

The repository includes:

* ✔️ Pretraining script (`Pretraining_gpt2.py`)
* ✔️ Inference script (`hf_inference.py`)
* ✔️ Custom datasets (`*.txt` files)
* ✔️ Example usage
* ✔️ Instructions to reproduce training & inference



## 📌 What This Model Does

This project performs **domain-adapted pretraining** (continued training) on GPT-2 using Hugging Face’s Transformers library.

After training, the model can:

* Generate text aligned with your datasets
* Answer questions about cricket, education, medical topics, etc.
* Produce GPT-2-style completions based on your training data
* Learn vocabulary and patterns found in your custom files

All trained model files are stored in:


trained_model/




## 📂 Project Structure



├── Cricket.txt
├── Education.txt
├── Medical.txt
├── Pretraining_gpt2.py
├── hf_inference.py
├── test_gpt2.py
└── trained_model/  (generated after training)




## ⚙️ Installation

First, install required dependencies:

bash
pip install transformers datasets accelerate safetensors


(Optional for GPU acceleration)

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121




## 🚀 Training the Model

Run the pretraining script:

bash
python Pretraining_gpt2.py


This script:

* Loads all `.txt` files in the current directory
* Tokenizes your data
* Trains GPT-2 using Causal Language Modeling
* Saves the final model to `trained_model/`

After training finishes, you will see:


trained_model/
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer.json
    ├── vocab.json
    └── merges.txt




## 🤖 Running Inference

Use `hf_inference.py` to generate text from your trained model.

### Run:

bash
python hf_inference.py


Make sure the script points to your trained model:

python
model_path = "trained_model"


You can modify the prompt inside the script to test different inputs.



## 📝 Example Inputs & Outputs

### Input:


Explain the basics of cricket batting:


### Output:


Cricket batting involves proper stance, balance, and footwork.
A batsman should watch the ball closely, judge the length early,
and choose between a defensive or attacking shot. Timing and practice
help improve consistency and shot selection.




### Input:


What is Artificial Intelligence?


### Output:


Artificial Intelligence refers to the ability of computers and machines
to perform tasks that typically require human intelligence, such as learning,
reasoning, decision making, and natural language understanding.




## 🧪 Testing the Model

You may use `test_gpt2.py` (or inference script) to try out various prompts and test generations.



## 📚 Technologies Used

* **Python**
* **Hugging Face Transformers**
* **Datasets Library**
* **PyTorch**
* **GPT-2**
