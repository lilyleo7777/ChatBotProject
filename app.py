import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_community.vectorstores import Pinecone as PineconeVectorStore


# Set up the environment variable for API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

#loading FAQ document 
loader = TextLoader("./ADHDFAQ.txt") 
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# Extract questions and answers
questions = []
qna_pairs = []

for doc in docs:
    if doc.page_content.startswith('Q:'):
        questions.append(doc.page_content.split('\n')[0])
        qna_pairs.append(doc)

#creating embeddings using HuggingFace
modelPath = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name_qa2 = 'adhd-qa2'

#connecting to index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name_qa2 not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name_qa2,
        dimension=1024,  # dimensionality of our embedding model
        metric='cosine',
        spec=spec
    )
    while not pc.describe_index(index_name_qa2).status['ready']:
        time.sleep(1)
index_qa2 = pc.Index(index_name_qa2)
time.sleep(1)

#creating vectorstore that holds FAQ doc embeddings 
text_field = "text"  # the metadata field that contains our text
vectorstore_qa2 = PineconeVectorStore(index_qa2, embedding_model.embed_query, text_field)

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

# Initialising context - System Message
context = [HumanMessage(content='Your role is an assistant for an ADHD specialist clinic called ADHD Specialists Australia to answer questions people have regarding the clinic and ADHD. You first welcome me and ask how you can help me.')]
response = model(context)
context.append(response)
st.markdown(response.content)

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

query = st.text_input("Your query:")

if query:
    human_message = HumanMessage(content=query)
    new_context = HumanMessage(content=augment_prompt_qa(query))
    context.append(new_context)
    context.append(human_message)
    # st.session_state.chat_history.append(human_message)

    response = model(context)
    context.append(response)
    # st.session_state.chat_history.append(response)

    st.markdown(response.content)

# for message in st.session_state.chat_history:
#     st.markdown(message.content)

