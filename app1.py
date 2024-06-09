import streamlit as st
import os
import google.generativeai as genai
import time
import pinecone
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec
# import logging
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
# #from langchain_community.vectorstores import Pinecone as PineconeVectorStore
#from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings


# Set up the environment variable for API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
embedding_model = load_embedding_model()
# q = "what is ADHD?"
# q_embedding = embedding_model.embed_query(q)

# # Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name_qa2 = 'adhd-qa2'
index_qa2 = pc.Index(index_name_qa2)

# queried = index_qa2.query(
#     vector = q_embedding,
#     top_k=3,
#     include_values=False,
#     include_metadata=True
# )
# top_3 = [match['metadata'].get('text', 'No text metadata found') for match in queried['matches']]

# time.sleep(1)

# #creating vectorstore that holds FAQ doc embeddings 
# text_field = "text"  # the metadata field that contains our text
# vectorstore_qa2 = PineconeVectorStore(index_qa2, embedding_model, text_field)

# chat model
model = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro-latest",
                               temperature=0.2, top_p=0.1)

# function that creates context from top 3 similarities 

def augment_prompt_qa(query):
    # get top 3 results from knowledge base
    q_embedding = embedding_model.embed_query(query)
    
    results = index_qa2.query(
            top_k=3,
            vector=q_embedding,
            include_values=False,
            include_metadata = True
        )
    
    top_3 = [match['metadata'].get('text', 'No text metadata found') for match in results['matches']]

    source_knowledge = '\n'.join(top_3)

    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# response = random.choice(
#         [
#             "Hello there! How can I assist you today?",
#             "Hi, human! Is there anything I can help you with?",
#             "Do you need help?",
#         ]
#     )
# Streamlit app layout
st.title("ADHD Specialist Australia Clinic Chatbot")

inital_system_message = "Your role is an assistant for an ADHD specialist clinic called \
ADHD Specialists Australia to answer questions people have regarding the clinic and ADHD. \
You first welcome me and ask how you can help me."

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "context" not in st.session_state:
    st.session_state.context = [HumanMessage(content = inital_system_message)]
    response = model(st.session_state.context)
    st.session_state.context.append(response)
    # with st.chat_message("assistant"):
    #     st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
        

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# st.write("DEBUG: st.session_state.context", st.session_state.context)

# Accept user input
if query := st.chat_input("What is your query?"):
    with st.chat_message("user"):
        st.markdown(query)
        
    new_context = HumanMessage(content=augment_prompt_qa(query)) 
    human_message = HumanMessage(content=query)
    # st.write("DEBUG: new context:", new_context)
    # st.write("DEBUG: system message:", human_message)
    # st.write("DEBUG: st.session_state.context", st.session_state.context)

    st.session_state.context.append(new_context)
    st.session_state.context.append(human_message)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # response = augment_prompt_qa(query)
    # st.write("DEBUG: Query:", query)
    # st.write("DEBUG: Response:", response)
        
    
    response = model(st.session_state.context)
    st.session_state.context.append(response)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    with st.chat_message("assistant"):
        for i in range(len(response.content)):
            st.markdown(response.content[:i+1])
            time.sleep(0.05)
        # st.markdown(response.content)
    # st.session_state.messages.append({"role": "assistant", "content": response})

