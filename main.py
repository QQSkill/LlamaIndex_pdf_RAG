from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pdf_engine import vietnam_engine
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()
HF_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

population_path = os.path.join('data', 'WorldPopulation2023.csv')
pop_df = pd.read_csv(population_path)

llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha", api_key=HF_TOKEN)
embed_model = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name='thenlper/gte-large'))

# result = llm.complete("Instruction: Just response the climate. Question: What is the climate of Viet Nam? Anwser:")
# print(result)
# completion_response = llm.complete("To infinity, and")
# print(completion_response)

# population_query_engine = PandasQueryEngine(df=pop_df, verbose=True, instruction_str=instruction_str, llm=llm)
# population_query_engine.update_prompts({"pandas_prompt": new_prompt})
# population_query_engine.query("what is population of Canada?")


# RAG with pdf
while True:
    prompt = input("Enter a Question about Viet Nam (q to quit): ")
    if prompt == 'q':
        break
    result = vietnam_engine.query(prompt)
    print(result)

# LLM Agent
# tools = [
#     note_engine,
#     QueryEngineTool(
#         query_engine=population_query_engine,
#         metadata=ToolMetadata(
#             name="population_data",
#             description="this gives information at the world population and demographics",
#         ),
#     ),
#     QueryEngineTool(
#         query_engine=vietnam_engine,
#         metadata=ToolMetadata(
#             name="pdf_query",
#             description="this gives detailed information about vietnam the country",
#         ),
#     ),
# ]

# agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# while True:
#     prompt = input("Enter a prompt (q to quit): ")
#     print(prompt)
#     if prompt == 'q':
#         break
#     result = agent.query(prompt)
#     print(result)