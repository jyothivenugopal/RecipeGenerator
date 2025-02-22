from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from os import environ


# Create an llm object to use for the QueryEngine and the ReActAgent
llm = OpenAI(model="gpt-4")
def load_index(recipe_persist_dir):
    storage_context = StorageContext.from_defaults(
            persist_dir=recipe_persist_dir
        )
    recipe_index = load_index_from_storage(storage_context)
    return recipe_index

def load_multiple_index(recipe_persist_dirs):
    recipe_indices = []
    for recipe_persist_dir in recipe_persist_dirs:
        recipe_indices.append(load_index(recipe_persist_dir))
    return recipe_indices

def build_rag_agent(recipe_persist_dir, k = 3, max_turns = 10):
    index = load_index(recipe_persist_dir)
    
    recipe_engine = index.as_query_engine(similarity_top_k=k, llm=llm)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=recipe_engine,
            metadata=ToolMetadata(
                name='Recipe',
                description=(
                    "Provides information about a list of recipe and ingredients! "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        )
    ]
    

    
    agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    max_turns=max_turns,
    )

    return agent

if __name__ == '__main__':
    import sys
    OPENAI_API_KEY =environ["OPENAI_API_KEY"]

    vector_store = sys.argv[1]
    chat_msg = sys.argv[2]
    agent = build_rag_agent(vector_store)
    response = agent.chat(chat_msg)
    print(response)



