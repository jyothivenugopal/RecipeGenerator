from os import environ
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

if __name__ == "__main__":

    load_dotenv()

    OPENAI_API_KEY =environ["OPENAI_API_KEY"]

    # generate vector store for a recipe
    # recipe_persist_dir = "./storage/test_recipe"
    # recipe_name = "test_recipe"
    recipe_persist_dir = sys.argv[1]
    recipe_name = sys.argv[2]
    input_files = sys.argv[3].split(',')

    assert all([Path(f).exists for f in input_files])


    # Create an llm object to use for the QueryEngine and the ReActAgent
    llm = OpenAI(model="gpt-4")

    
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=recipe_persist_dir
        )
        recipe_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        # load data
        recipe_docs = SimpleDirectoryReader(
            input_files=input_files
        ).load_data()
        

        # build index
        recipe_index = VectorStoreIndex.from_documents(recipe_docs, show_progress=True) #local to bypass
        
        # persist index
        recipe_index.storage_context.persist(persist_dir=recipe_persist_dir)

    recipe_engine = recipe_index.as_query_engine(similarity_top_k=3, llm=llm)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=recipe_engine,
            metadata=ToolMetadata(
                name=recipe_name,
                description=(
                    "Provides information about a list of recipe and ingredients! "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        )
    ]
