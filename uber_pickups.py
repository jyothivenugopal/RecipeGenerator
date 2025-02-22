import streamlit as st
import pandas as pd
import numpy as np
import openai
import os
from pinecone import Pinecone, ServerlessSpec

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def get_recipe(ingredients):
    prompt = f"I have {ingredients}. What can I cook?"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response.choices[0].message.content


st.title('Recipe Recommender')
st.write("Enter your ingredients and get recipe suggestions!")

# User input
ingredients = st.text_input("Enter ingredients (comma-separated):", "")

# print recipe on the UI
if st.button("Generate Recipe"):
    if ingredients:
        with st.spinner("Generating recipe..."):
            recipe = get_recipe(ingredients)
            st.subheader("Here’s what you can cook:")
            st.write(recipe)
    else:
        st.warning("Please enter some ingredients first!")

# Load dataset (update filename accordingly)
# df = pd.read_csv("RecipeNLG_dataset.csv")

#Pinecone

index_name = "recipe-indexjv"
pc.create_index(
    name=index_name,
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# Connect to the index
index = pc.Index(index_name)

# Load Kaggle dataset
df = pd.read_csv("RecipeNLG_dataset.csv")  # Update with your dataset path

# Function to generate embeddings from text
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Preview dataset
# print(df.head())