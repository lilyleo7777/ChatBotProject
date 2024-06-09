import streamlit as st
import os
import pinecone
import google.generativeai as genai
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from langchain_pinecone import Pinecone   # Updated import
from langchain_pinecone import PineconeVectorStore 

# Set up the environment variable for API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Pinecone
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"))

index_name_qa2 = 'adhd-qa2'

index_qa2 = pinecone.Index(index_name_qa2)
time.sleep(1)

#creating vectorstore that holds FAQ doc embeddings 
text_field = "text"  # the metadata field that contains our text
vectorstore_qa2 = Pinecone(index_qa2, embedding_model, text_field)

# chat model
model = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro-latest",
                               temperature=0.3, top_p=0.2)

#function that creates context from top 3 similarities 
def augment_prompt_qa(query: str):
    # get top 3 results from knowledge base
    results = vectorstore_qa2.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


# Streamlit app layout
st.title("ADHD Specialist Australia Clinic Chatbot")

inital_system_message = "Your role is an assistant for an ADHD specialist clinic called \
ADHD Specialists Australia to answer questions people have regarding the clinic and ADHD. \
You first welcome me and ask how you can help me."


if "context" not in st.session_state:
    st.session_state.context = [HumanMessage(content = inital_system_message)]
    response = model(st.session_state.context)
    st.session_state.context.append(response)
    with st.chat_message("Assistant"):
        st.markdown(response.content)
        
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("What is your query?"):
  human_message = HumanMessage(content=query)
  new_context = HumanMessage(content=augment_prompt_qa(query))
  st.session_state.context.append(new_context)
  st.session_state.context.append(human_message)
  st.session_state.messages.append({"role": "user", "content": query})

  # Display user message in chat message container
  with st.chat_message("user"):
    st.markdown(query)

  response = model(st.session_state.context)

  with st.chat_message("assistant"):
    for word in response.content.split():
      st.markdown(word + " ")
      time.sleep(0.05)
