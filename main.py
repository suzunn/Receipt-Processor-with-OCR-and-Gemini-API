from PIL import Image
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# API anahtarlarını al
ocr_api_key = os.getenv('OCR_API_KEY')
if not ocr_api_key or not os.getenv("GEMINI_API_KEY"):
    raise ValueError("API anahtarları .env dosyasından yüklenemedi.")

# OCR.space API URL'si
url = 'https://api.ocr.space/parse/image'

# Görsel dosyasını aç
image = Image.open('receipt.jpg')

# Görseli yeniden boyutlandırarak kaydet
image = image.resize((image.width // 2, image.height // 2))  # Boyutu yarıya indir
image.save('resized_image.jpg', quality=85)  # Sıkıştırarak kaydet

# Görsel dosyanızın yolu
image_path = 'resized_image.jpg'

# Görseli binary formatında okuma
with open(image_path, 'rb') as image_file:
    # API'ye gönderilecek veriler
    payload = {
        'apikey': ocr_api_key,
        'language': 'tur',  # Dil seçeneklerini buradan değiştirebilirsiniz (örneğin, 'tur' Türkçe için)
        'isTable': 'true',  # Tablo tanıma yapmak için bu parametreyi kullanabilirsiniz
        'OCREngine': 2,  # OCR Engine 2'yi kullan
        'detectOrientation': 'true',  # Görselin yönünü otomatik algıla
    }
    
    # Görseli ve diğer parametreleri API'ye gönder
    files = {
        'file': image_file
    }
    
    # POST isteğini gönder
    response = requests.post(url, data=payload, files=files)
    
    # Yanıtı kontrol et
    if response.status_code == 200:
        result = response.json()
        
        if 'ParsedResults' not in result:
            print("OCR yanıtında 'ParsedResults' bulunamadı.")
            print("Yanıt: ", result)
            exit(1)
        
        ocr_text = result['ParsedResults'][0]['ParsedText']
        print("OCR ile çıkarılan metin: ", ocr_text)

        # Gemini Prompt
        prompt = f"""
        We are processing a receipt, which may have irregular formatting or noise in the text. Please extract the following details from the receipt below, ensuring the response is clear and correctly formatted:

        1. **Market Name**: The name of the market/store.
        2. **Date**: The date the receipt was issued. Include the day, month, and year.
            3. **Time**: The time the receipt was issued, in a 24-hour or 12-hour format (e.g., 12:55).
        4. **City**: The city where the receipt was issued.
        5. **Total Price**: The total amount on the receipt. If there are extra spaces or unnecessary symbols (e.g., commas, dots), they should be removed. Only the final price should be returned with "TL" appended at the end, and if there are any kuruş values, they should be included (e.g., 100.75 TL).
        6. **Items**: A list of items purchased along with their individual prices. The item names should be clear, and the prices should be in the correct format (numbers only, with "TL" at the end if needed).

        Additionally, please include the **name of the receipt holder** (if available) in the response.

        Format the result strictly as a JSON object with the following fields:
        - "market": (string) Name of the market/store.
        - "date": (string) The date of the receipt.
        - "time": (string) The time the receipt was issued.
        - "city": (string) The city where the receipt was issued.
        - "total": (string) The total price formatted correctly.
        - "items": (array of objects) A list of items, where each item is represented as:
           - "name": (string) The name of the item.
            - "price": (string) The price of the item, formatted correctly (e.g., "20.50 TL").

        Receipt text:
        {ocr_text}

        """

        # Gemini API modelini kullan
        model = genai.GenerativeModel("gemini-2.0-flash")
        gemini_response = model.generate_content(prompt)

        # Gemini API yanıtını yazdır
        print("Gemini API yanıtı: ", gemini_response.text)
    else:
        print("OCR.space API Hatası:", response.status_code, response.text)