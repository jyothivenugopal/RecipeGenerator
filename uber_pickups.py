import streamlit as st
import pandas as pd
import numpy as np
import openai
import os
from pinecone import Pinecone, ServerlessSpec
import time

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#def get_recipe(ingredients):
#     prompt = f"I have {ingredients}. What can I cook?"
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#     )
    
#     return response.choices[0].message.content

# Load dataset (update filename accordingly)
# df = pd.read_csv("RecipeNLG_dataset.csv")

#Pinecone

index_name = "recipe-indexjv"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

# Wait for the index to be ready
# while not pc.describe_index(index_name).status['ready']:
#     time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Load Kaggle dataset
df = pd.read_csv("subset.csv")  # Update with your dataset path

# Function to generate embeddings from text
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to process DataFrame and store recipes in Pinecone
def store_kaggle_recipes():
    vectors = []
    
    for idx, row in df.iterrows():
        # Create a combined text description for embedding
        recipe_text = f"{row['title']}: Ingredients - {row['ingredients']}. Instructions - {row['directions']}"
        # Generate embedding for the recipe
        embedding = get_embedding(recipe_text)
        
        # Append to batch
        vectors.append((
            f"recipe_{idx}",  # Unique ID
            embedding,  # Vector representation
            {"text": recipe_text, "diet": row.get("Diet", "Unknown")}
        ))

        # Insert in batches of 100 (to optimize API usage)
        if len(vectors) >= 100:
            index.upsert(vectors=vectors)
            vectors = []  # Reset batch

    # Insert any remaining data
    if vectors:
        index.upsert(vectors=vectors)

    print(f"{len(df)} recipes successfully stored in Pinecone!")

# Run once to populate the index
# store_kaggle_recipes()

def search_recipes(user_input):
    query_vector = get_embedding(user_input)
    
    results = index.query(
        vector=query_vector,
        top_k=3,  # Get top 3 similar recipes
        include_metadata=True
    )

    # Prepare the results as a list of dictionaries
    recipes = [
        {
            "recipe": match["metadata"].get("text", "No recipe text available"),  # Safe access
            "score": match.get("score", 0),  # Default score if missing
        }
        for match in results.get("matches", [])
    ]
    
    return recipes

st.title('Recipe Recommender')
st.write("Enter your ingredients and get recipe suggestions!")

# User input
ingredients = st.text_input("Enter ingredients (comma-separated):", "")

# print recipe on the UI
if st.button("Generate Recipe"):
    if ingredients:
        with st.spinner("Generating recipe..."):
            recipes = search_recipes(ingredients)
            
            st.subheader("Here’s what you can cook:")
            
            # Loop through the recipes and display them in a clean format
            for idx, recipe in enumerate(recipes, start=1):
                st.markdown(f"**Recipe {idx}:**")
                st.markdown(f"**Ingredients**: {ingredients}")
                st.markdown(f"**Recipe Details**: {recipe['recipe']}")
                st.markdown(f"**Score**: {recipe['score']:.2f}")
                st.write("---")  # Add a separator between recipes
    else:
        st.warning("Please enter some ingredients first!")

# Preview dataset
# print(df.head())