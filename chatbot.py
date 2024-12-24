import openai
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import time

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    exit()
openai.api_key = openai_api_key

try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print("test")

except Exception as e:
    print(f"Error downloading NLTK resources: {str(e)}")
    exit()

def preprocess_text(text):
    try:
        if not text:
            print("Error: No text provided.")
            return ""
        
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)

        if not words:
            print("Error: Tokenization failed, no words found.")
            return ""

        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)

    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return ""

def chatbot_test(user_input):
    try:
        start_time = time.time()
        cleaned_input = preprocess_text(user_input)

        if not cleaned_input:
            print("Error: Preprocessed input is empty. Check your input or preprocessing.")
            return

        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": cleaned_input}
            ]
        )
        bot_response = response['choices'][0]['message']['content'] 
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Response: {bot_response}")
        print(f"Processing Time: {processing_time:.2f} seconds")

    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {str(e)}")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    try:
        user_input = input("You: ")
        chatbot_test(user_input)
        
    except Exception as e:
        print(f"Error in chatbot test: {str(e)}")
