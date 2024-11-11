import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from googletrans import Translator
# Tải mô hình CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
# Tải pipeline phân loại toxic comments
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
# Danh sách nhãn nhạy cảm
sensitive_labels = [
    "toxic", "nudity", "violence", "horror", "blood", "gore", "murder", 
    "assault", "abuse", "self-harm", "slasher", "disturbing", "graphic", "cruelty", 
    "hate", "terrorism", "suicide", "death", "rape", "torture", "war", "execution",
    "drugs", "child abuse", "weapon", "stabbing", "shooting", "massacre", "genocide", 
    "animal cruelty", "domestic violence", "bullying", "abortion", "explosion", "poison",
    "addiction", "gang violence", "extremism", "hostage", "harassment", "racism", 
    "sex trafficking", "human trafficking", "human rights violations", "sexually explicit", 
    "extreme violence", "sexual abuse", "brutality", "crimes against humanity", "child pornography"
]

# Hàm tải ảnh từ URL
def load_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        if 'image' not in response.headers['Content-Type']:
            return None

        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
            requests.exceptions.Timeout, requests.exceptions.RequestException,
            UnidentifiedImageError) as e:
        print(f"Error loading image: {e}")
        return None

# Hàm kiểm tra ảnh nhạy cảm
def check_sensitive_image(image_url):
    img = load_image(image_url)
    if img is None:
        print("Không thể tải ảnh hoặc ảnh không hợp lệ.")
        return False, None

    inputs = clip_processor(text=sensitive_labels, images=img, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    for i, label in enumerate(sensitive_labels):
        if probs[0][i] > 0.5:
            print(f"Ảnh nhạy cảm phát hiện: {label} (độ tin cậy: {probs[0][i]:.2f})")
            return True, label

    print("Ảnh không nhạy cảm.")
    return False, None
# Tải pipeline phân loại toxic comments
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")

# Dịch văn bản từ tiếng Việt sang tiếng Anh
def translate_vietnamese_to_english(vietnamese_text):
    translator = Translator()
    translation = translator.translate(vietnamese_text, src='vi', dest='en')
    return translation.text
# Hàm kiểm tra xem văn bản có phải là tiếng Việt hay không
def is_vietnamese(text):
    # Kiểm tra nếu văn bản có chứa ký tự tiếng Việt
    return any(ord(c) > 127 for c in text)
# Hàm phát hiện toxicity
def check_sensitive_text(text):
    # Nếu văn bản là tiếng Việt, dịch sang tiếng Anh
    if is_vietnamese(text):
        text = translate_vietnamese_to_english(text)
    
    # Phân loại toxicity và giả sử toxicity_classifier trả về một dictionary với 'score'
    result = toxicity_classifier(text)
    
    # Lấy điểm toxicity từ kết quả trả về (dưới dạng dictionary)
    toxicity_score = result[0]['score'] if isinstance(result, list) else result['score']
    
    # In kết quả toxicity ra (nếu cần)
    print(toxicity_score)
    
    # Kiểm tra nếu độ độc hại cao hơn 0.8
    if toxicity_score > 0.5:
        return True
    else:
        return False

