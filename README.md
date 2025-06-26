## 📥 Download Pretrained Models

Before running the code, please manually download the required pretrained components from Hugging Face and place them in the following directory structure:


### 🔗 Download Links:
You can get them from the [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
pretrained_model/
└── stable-diffusion-v1-4/
    ├── text_encoder/
    ├── tokenizer/
    └── scheduler/

### 📌 Usage in Code:
Make sure your model paths are set like this:
```python
text_encoder = CLIPTextModel.from_pretrained("pretrained_model/stable-diffusion-v1-4/text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("pretrained_model/stable-diffusion-v1-4/tokenizer")
scheduler = DDPMScheduler.from_pretrained("pretrained_model/stable-diffusion-v1-4/scheduler")
