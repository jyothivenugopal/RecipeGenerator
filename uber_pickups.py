import streamlit as st
import pandas as pd
import numpy as np
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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