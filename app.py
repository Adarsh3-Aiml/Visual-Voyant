from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from io import BytesIO


app = Flask(__name__)

# Load the pre-trained Vilt model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image and question from the form
        image_file = request.files["image"]
        question = request.form["question"]

        # Check if the user provided an image file and a question
        if image_file and question:
            # Open and process the image using Pillow
            image = Image.open(image_file)
            image_format = image.format
            if image_format not in ["JPEG", "PNG"]:
                # Convert the image to JPEG or PNG
                image = image.convert("JPEG")  # or "PNG"

            # Ensure the image is in RGB format
            image = image.convert("RGB")

            # Perform VQA
            encoding = processor(image, question, return_tensors="pt")
            outputs = model(**encoding)
            logits = outputs.logits

            # Get the predicted answer
            predicted_answer_id = logits.argmax(-1).item()
            predicted_answer = processor.tokenizer.decode(predicted_answer_id)

            return render_template("result.html", image=image, question=question, answer=predicted_answer)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)