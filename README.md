The Recipe generator is a RAG AI agent that outputs recipes that you can prepare with ingredients that are in your fridge. Just upload a picture of your refrigerator, and it will suggest different recipes you can cook.
It outputs suggestions from ChatGPT and a vector database which has embeddings stored into it from a Recipe Dataset from Kaggle. I have also used Pinecone to create the index which has the embeddings and query against
the index based on user query. I have created the embeddings using OpenAI's API.
