__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st

#Importations
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, tempfile, glob, random
from pathlib import Path
from getpass import getpass
import numpy as np
from itertools import combinations
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory

# LLM: openai and google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM: HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.schema import Document, format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Chroma: vectorstore
from langchain_community.vectorstores import Chroma

# Contextual Compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("Documents")
documents = loader.load()

# Create a RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " ", ""],
    chunk_size = 1600,
    chunk_overlap= 200
)

# Text splitting
chunks = text_splitter.split_documents(documents=documents)
print(f"number of chunks: {len(chunks)}")

def select_embeddings_model(LLM_service="HuggingFace"):
    """Connect to the embeddings API endpoint by specifying the name of the embedding model."""
    if LLM_service == "OpenAI":
        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_key=openai_api_key)

    if LLM_service == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyCMA5G8t2UnD5q92arsYrwAbSm19xZlZV4",
            timeout=300
        )
    if LLM_service == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key="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",
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )

    return embeddings

embeddings_google = select_embeddings_model(LLM_service="HuggingFace")
#embeddings_HuggingFace = select_embeddings_model(LLM_service="HuggingFace")

def create_vectorstore(embeddings,documents,vectorstore_name):
    """Create a Chroma vector database."""
    LOCAL_VECTOR_STORE_DIR = Path("vector_store")
    persist_directory = (LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + vectorstore_name)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


vector_store_google = create_vectorstore(
        embeddings=embeddings_google,
        documents = chunks,
        vectorstore_name="Ifri"
    )
print("vector_store_google:",vector_store_google._collection.count(),"chunks.")

def Vectorstore_backed_retriever(vectorstore,search_type="similarity",k=5,score_threshold=None):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever


base_retriever_google = Vectorstore_backed_retriever(vector_store_google,"similarity",k=5)

def instantiate_LLM(LLM_provider="HuggingFace",api_key="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",temperature=0.5,top_p=0.95,model_name=None):
    """Instantiate LLM in Langchain.
    Parameters:
        LLM_provider (str): the LLM provider; in ["OpenAI","Google","HuggingFace"]
        model_name (str): in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo-preview",
            "gemini-pro", "mistralai/Mistral-7B-Instruct-v0.2"].
        api_key (str): google_api_key or openai_api_key or huggingfacehub_api_token
        temperature (float): Range: 0.0 - 1.0; default = 0.5
        top_p (float): : Range: 0.0 - 1.0; default = 1.
    """
    if LLM_provider == "OpenAI":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            model_kwargs={
                "top_p": top_p
            }
        )
    if LLM_provider == "Google":
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            # model="gemini-pro",
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            convert_system_message_to_human=True
        )
    if LLM_provider == "HuggingFace":
        llm = HuggingFaceHub(
            # repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature":temperature,
                "top_p": top_p,
                "do_sample": True,
                "max_new_tokens":1024
            },
        )
    return llm

def create_memory(model_name='gpt-3.5-turbo',memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models."""

    if model_name=="gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024 # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=openai_api_key,temperature=0.1),
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question"
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )
    return memory


def answer_template():
    """Pass the standalone question along with the chat history and context (retrieved documents) to the `LLM` to get an answer."""

    template = f"""Vous êtes un assistant virtuel spécialisé dans l'Institut de formation et de recherche en Informatique. Votre rôle est de répondre aux questions des utilisateurs de manière précise et informative, en vous basant uniquement sur le contexte fourni ci-dessous. 

### Instructions :
1. **Contexte** : Utilisez uniquement les informations contenues dans le bloc <context></context> pour formuler votre réponse. Ne faites pas d'hypothèses ou de généralisations en dehors de ce qui est fourni.
   
2. **Clarté et Concision** : Votre réponse doit être claire, concise et directement liée à la question posée. Évitez les réponses vagues ou trop longues.

3. **Langue** : Formulez votre réponse dans la langue spécifiée à la fin de la question.

4. **Pertinence** : Assurez-vous que votre réponse est pertinente par rapport à l'historique de la conversation et au contexte fourni. Si vous ne trouvez pas d'information pertinente, indiquez-le clairement.

### Contexte :
<context>
{{chat_history}}

{{contexte}}
</context>

### Question :
{{question}} 

"""
    return template

def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def custom_ConversationalRetrievalChain(
    llm,condense_question_llm,
    retriever,
    language="english",
    llm_provider="OpenAI",
    model_name='gpt-3.5-turbo',
):
    """Create a ConversationalRetrievalChain step by step.
    """
    ##############################################################
    # Step 1: Create a standalone_question chain
    ##############################################################

    # 1. Create memory: ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models

    memory = create_memory(model_name)
    # memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)

    # 2. load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )

    # 3. Pass the follow-up question along with the chat history to the LLM, and parse the answer (standalone_question).

    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'],
        template = """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""
)

    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_question_llm
        | StrOutputParser(),
    }

    # 4. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain

    ####################################################################################
    #   Step 2: Retrieve documents, pass them to the LLM, and return the response.
    ####################################################################################

    # 5. Retrieve relevant documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # 6. Get variables ['chat_history', 'context', 'question'] that will be passed to `answer_prompt`

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))
    # 3 variables are expected ['chat_history', 'context', 'question'] by the ChatPromptTemplate
    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history") # get it from `loaded_memory` variable
    }

    # 7. Load memory, format `answer_prompt` with variables (context, question and chat_history) and pass the `answer_prompt to LLM.
    # return answer, docs and standalone_question

    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata
        "docs": lambda x: [Document(page_content=doc.page_content,metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x:x["question"] # return standalone_question
    }

    # 8. Final chain
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer

    print("Conversational retriever chain created successfully!")

    return conversational_retriever_chain,memory


chain_gemini,memory_gemini = custom_ConversationalRetrievalChain(
    llm = instantiate_LLM(),
    condense_question_llm = instantiate_LLM(),
    retriever=base_retriever_google
)


with st.sidebar:
    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat",key=reset_button_key)
    if reset_button:
        st.session_state.conversation = None
        st.session_state.chat_history = None
    "[View the source code](https://github.com/Arix-ALIMAGNIDOKPO/IFRI-AI-Chatbot)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Arix-ALIMAGNIDOKPO/IFRI-AI-Chatbot)"
st.image('ifri.png', width=200)
st.caption("🚀 Bienvenue sur le chatbot de l'IFRI !")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "En quoi pouvons nous vous aider ?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Afficher un spinner pendant le traitement de la question
    with st.spinner('Traitement de la question en cours...'):
        response = chain_gemini.invoke({"question":prompt})
        
    msg = response['answer'].content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    
    # Afficher un spinner pendant la génération de la réponse    
    with st.spinner('Génération de la réponse en cours...'):
        st.chat_message("assistant").write(msg)
        
    memory_gemini.save_context({"question": prompt}, {"answer": msg})
