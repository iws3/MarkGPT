# Deployment Guide

Deploy MarkGPT model to production.

## Local Deployment (Flask)

```python
from flask import Flask, jsonify, request
from src.model.markgpt import MarkGPT
import torch

app = Flask(__name__)
model = MarkGPT.load_pretrained("checkpoints/final.pt")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    tokens = tokenizer.encode(prompt)
    generated = model.generate(tokens, max_length=100)
    return jsonify({"text": tokenizer.decode(generated)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

## Cloud Deployment (AWS EC2)

```bash
# Launch instance
aws ec2 run-instances --image-id ami-12345 --instance-type g4dn.xlarge

# SSH and deploy
ssh -i key.pem ubuntu@instance-ip
git clone repo
pip install -r requirements.txt
python deploy/app.py  # Start server
```

## Containerized Deployment (Docker)

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "deploy/app.py"]
```

```bash
docker build -t markgpt:latest .
docker run -p 5000:5000 markgpt:latest
```

## Hugging Face Model Hub

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model.push_to_hub("yourusername/MarkGPT")
tokenizer.push_to_hub("yourusername/MarkGPT")
```

Users can then:
```python
model = AutoModelForCausalLM.from_pretrained("yourusername/MarkGPT")
```

