import streamlit as st
import pandas as pd
import numpy as np

st.title('Recipe Recommender')

st.write("Enter your ingredients and get recipe suggestions!")

# User input
ingredients = st.text_input("Enter ingredients (comma-separated):", "")