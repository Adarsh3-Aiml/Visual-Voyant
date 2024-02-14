import requests
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

def run():
    url = "https://images.pexels.com/photos/1996332/pexels-photo-1996332.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "How many horses are there?"
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])

if __name__ == "__main__":
    run()