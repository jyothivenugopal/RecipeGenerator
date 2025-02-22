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

def get_recipe_from_agent(ingredients):
    from agent import build_rag_agent
    recipe_persist_dir = "vector_store_generation/storage/test_recipe"
    agent = build_rag_agent(recipe_persist_dir)
    response = agent.chat(f'I have the following {ingredients}  What should I make?')
    return response

st.title('Recipe Recommender')
st.write("Enter your ingredients and get recipe suggestions!")

# User input
ingredients = st.text_input("Enter ingredients (comma-separated):", "")

# print recipe on the UI
if st.button("Generate Recipe"):
    if ingredients:
        with st.spinner("Generating recipe..."):
            gpt_recipe = get_recipe(ingredients)
            agent_recipe = get_recipe_from_agent(ingredients)
            st.subheader("Here’s what you can cook:")
            st.write(f"""
                     ChatGPT suggested:
                     {gpt_recipe}

                     Agent with a vector store suggested:
                     {agent_recipe}
                     """)
    else:
        st.warning("Please enter some ingredients first!")