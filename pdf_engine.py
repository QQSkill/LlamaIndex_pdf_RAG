import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
from llama_index.core import ServiceContext

load_dotenv()
HF_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha", api_key=HF_TOKEN)
embed_model = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name='thenlper/gte-large'))


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=512)
        index = VectorStoreIndex.from_documents(data, show_progress=True, embed_model=embed_model, service_context=service_context)
        index.storage_context.persist(persist_dir=index_name)
    else:
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=512)
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name), service_context=service_context
        )

    return index


pdf_path = os.path.join("data", "Vietnam.pdf")
vietnam_pdf = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
vietnam_index = get_index(vietnam_pdf, "vietnam")
vietnam_engine = vietnam_index.as_query_engine()