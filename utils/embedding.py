import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(chunks):
    return [client.embeddings.create(input=[chunk], model="text-embedding-3-small").data[0].embedding for chunk in chunks]
