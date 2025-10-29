import base64
import requests
import json
import pickle
from PIL import Image
from io import BytesIO
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Load dataset
_, _, test_data = pickle.load(open("datasets/promap_data.pickle", "rb"))

text_pairs = test_data["text_pairs"]
image_pairs = test_data["image_pairs"]
labels = test_data["labels"]

# Read API key from file
with open('.kiara-api-key', 'r') as f:
    api_key = f.read().strip()

# API endpoint and headers
url = "https://kiara.sc.uni-leipzig.de/api/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Results
y_true = []
y_pred = []
api_times = []

# Loop through test data
for i in range(len(text_pairs)):
    text_pair = text_pairs[i]
    image_pair = image_pairs[i]
    label = labels[i]

    # Open the two images
    img1 = Image.open(image_pair[0])
    img2 = Image.open(image_pair[1])

    #  Match heights of both images while preserving aspect ratio
    target_height = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * target_height / img1.height), target_height), Image.LANCZOS)
    img2 = img2.resize((int(img2.width * target_height / img2.height), target_height), Image.LANCZOS)

    # Combine images
    total_width = img1.width + img2.width
    combined_img = Image.new("RGB", (total_width, target_height))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    # Scale down to fit within max_size (512x512) if needed
    max_size = 512
    scale_factor = min(max_size / combined_img.width, max_size / combined_img.height, 1.0)
    if scale_factor < 1.0:  # Only resize if bigger than max
        new_width = int(combined_img.width * scale_factor)
        new_height = int(combined_img.height * scale_factor)
        combined_img = combined_img.resize((new_width, new_height), Image.LANCZOS)

    combined_img.save("debug.png", format="PNG")

    # Encode to Base64
    buffer = BytesIO()
    combined_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Generate prompt
    payload = {
        "model": "vllm-llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text":
                    "You are an expert in matching products from different online shops using both their titles and images.\n"
                    "The image you see contains two products side by side:\n"
                    "- The left product corresponds to the first title.\n"
                    "- The right product corresponds to the second title.\n"
                    f"First title (left image): {text_pair[0]}\n"
                    f"Second title (right image): {text_pair[1]}\n"
                    "Determine if these two offerings are the exact same real-world product.\n"
                    "- A 'match' means they are the same product and the same variant (color, size, material, etc.).\n"
                    "- A 'negative' means they are different products or different variants.\n"
                    "Output only:\n"
                    "1 → if they are the same product.\n"
                    "0 → if they are not the same product.\n"
                    "Do NOT output any explanation, punctuation, or whitespace.\n"
                 },
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }}
             ]}
        ],
    }

    # API request
    api_start = time.perf_counter()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    api_end = time.perf_counter()
    api_times.append(api_end - api_start)

    if response.ok:
        content = response.json()['choices'][0]['message']['content']

        # Save results
        try:
            pred = int(content.strip())
            y_true.append(label)
            y_pred.append(pred)

        except ValueError:
            print("Not a number:", content)

# Output metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

avg_request_time = (sum(api_times) / len(api_times)) if api_times else 0.0

print(f"Average request time:{avg_request_time:.3f} s")
