## Integrate our code OpenAI API
from operator import attrgetter
import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.document_loaders import AzureBlobStorageContainerLoader
import pathlib

import streamlit as st
from azure.storage.blob import BlobServiceClient

os.environ["OPENAI_API_KEY"]='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'
def celebrity_search(query):
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = False

    # Load the model
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        #loader = DirectoryLoader("data/")
        connection_string = "DefaultEndpointsProtocol=https;AccountName=sauravblob;AccountKey=gjpEChXv103U4yUKnQmnN8sGyX+yf/ZsZ9dBwy7yhQhARSJgBQPF2Ys9/i7NfYyiEXzG56ADojT++AStazjlvQ==;EndpointSuffix=core.windows.net"
        container_name = "sauravcontainer"

        loader = AzureBlobStorageContainerLoader(conn_str=connection_string, container=container_name)

        #data = loader.load()
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    # Create the conversational retrieval chain
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model="gpt-3.5-turbo"),
    #     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 5}),
    # )

    data = index.query_with_sources(query)
    #data = index.query_with_sources(query, k=5, model="gpt-3.5-turbo")
    #index.vectorstore.search_kwargs = {"k": 5}
    #data = index.query_with_sources(query,llm=ChatOpenAI(model="gpt-3.5-turbo"))
    data = index.query_with_sources(query,llm=OpenAI(temperature=0))
   
   
    print(data)

    #data = index.query_with_sources(query)
    # Get the query result
    #result = chain({"question": query, "chat_history": []})
    #print(result)
              
    return data

# Create a Streamlit app
st.title('GPT Smart Search Engine')

# Get the user input
input_text = st.text_input("Ask a question to your enterprise data lake")

# If the user enters a query, run the celebrity search function
if input_text:  
    data = celebrity_search(input_text)
    
    files = ''
    for filedata in data['sources'].split(','):
      print('file name',filedata)

      file_extension = pathlib.Path(filedata).suffix
      if file_extension:
        files = 'https://sauravblob.blob.core.windows.net/sauravcontainer/'+os.path.basename(filedata)
        st.write(files)
      else:
        files = filedata
        st.write(files)  
    
    st.write(data['answer'])


