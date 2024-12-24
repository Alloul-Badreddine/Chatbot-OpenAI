import openai
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import time

# Load environment variables
load_dotenv()

# Ensure environment variable is loaded correctly
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    exit()

# Set OpenAI API Key
openai.api_key = openai_api_key

# Preload NLTK resources and handle errors
try:
    # Download necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading NLTK resources: {str(e)}")
    exit()

# Define text preprocessing function
def preprocess_text(text):
    try:
        if not text:
            print("Error: No text provided.")
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize words using explicit punkt tokenizer
        words = word_tokenize(text)

        if not words:
            print("Error: Tokenization failed, no words found.")
            return ""

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return ""

# Function to test chatbot directly
def chatbot_test(user_input):
    try:
        start_time = time.time()

        # Preprocess input for faster text processing
        cleaned_input = preprocess_text(user_input)

        if not cleaned_input:
            print("Error: Preprocessed input is empty. Check your input or preprocessing.")
            return

        # Generate response using OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": cleaned_input}
            ]
        )

        # Extract AI's reply
        bot_response = response['choices'][0]['message']['content']

        end_time = time.time()
        processing_time = end_time - start_time

        # Print response with processing time
        print(f"Response: {bot_response}")
        print(f"Processing Time: {processing_time:.2f} seconds")

    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

# Test the chatbot directly
if __name__ == '__main__':
    try:
        user_input = input("You: ")
        chatbot_test(user_input)
    except Exception as e:
        print(f"Error in chatbot test: {str(e)}")
