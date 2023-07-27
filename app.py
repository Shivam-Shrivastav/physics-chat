
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import pinecone 
from langchain.vectorstores import Pinecone
import streamlit as st
from streamlit_chat import message
import os
import random
from gtts import gTTS
from io import BytesIO
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
sound_file = BytesIO()
from langchain.prompts import PromptTemplate
from streamlit_option_menu import option_menu
from langchain.chains import LLMChain
import requests
from streamlit_lottie import st_lottie


os.environ["OPENAI_API_KEY"] = "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" 
EXAMPLE_NO = 1



st.set_page_config(page_title="Physics Buddy Chatbot ",
                       page_icon=":books:")
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Chat (QNA)", "Question Generator"],  # required
                icons=["chat", "question", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)


embeddings = OpenAIEmbeddings(openai_api_key='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', model='text-embedding-ada-002')

pinecone.init(
    api_key= '8c2cf921-9a39-469b-88e3-3e49ae8a927e',  # find at app.pinecone.io
    environment= 'us-west4-gcp'  # next to api key in console
)


template1 = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
{context}

Question: {question}
"""

template2 = """Create a list of questions which are generally ask in Physics Class 11 exam from {topic}
Dont answer like As an AI language model, I do not have personal experience or knowledge about Physics exam questions."""



PROMPT1 = PromptTemplate(template=template1, input_variables=["context", "question"])

PROMPT2 = PromptTemplate(template=template2, input_variables=["topic"])






@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def get_conversation_chain(PROMPT):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    vectorstore = Pinecone.from_existing_index('physics-chatbot', embeddings)

    memory = ConversationBufferMemory(
        output_key='answer',
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=800, temperature = 0.0),
        retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k":3}),
        memory=memory,
        chain_type="stuff", 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return conversation_chain


def get_summary(PROMPT):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    vectorstore = Pinecone.from_existing_index('physics-chatbot', embeddings)

    memory = ConversationBufferMemory(
        output_key='answer',
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=800, temperature = 0.0),
        retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k":3}),
        memory=memory,
        chain_type="stuff", 
        # return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return conversation_chain

def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()



if selected == "Chat (QNA)":
    # st.title(f"You have selected {selected} + 🤖 ")
    lottie_coding = load_lottieurl("https://lottie.host/58966f5f-bf71-443e-8492-04d245623324/IUMhRzvebv.json")

    with st.container():
        left, right = st.columns(2)
    with left:
        st.title(""" 

        Physics Buddy Chatbot
        
        """)
    with right:
        st_lottie(lottie_coding, height = 120)

    conversation_chain = get_conversation_chain(PROMPT1)

    # @st.cache
    def get_text():
        input_text = st.text_input("Enter Your Query Here: ","")
        return input_text 

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if 'generated' not in st.session_state:
        st.session_state.generated = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


    for message_, generated_ in zip(st.session_state.message_history, st.session_state.generated):
        try:
            message(message_,is_user=True, avatar_style = "adventurer", seed= "Sassy") # display all the previous message
            message(generated_, avatar_style="bottts", seed = "Milo")
        except Exception as e:
            print(e)
            if "please pass a unique `key`" in str(e):
                message(message_,is_user=True, avatar_style = "adventurer", seed= "Sassy") # display all the previous message
                message("Please ask the question in different way.", avatar_style="bottts", seed = "Milo")


    # # for generated_ in st.session_state.generated:
    # #     message(generated_)


    placeholder = st.empty() # placeholder for latest message
    input_ = get_text()
    # input_ = f'{input_} + in 100 words'
    if len(st.session_state.message_history) > 0 and input_ == st.session_state.message_history[-1]:
        pass
    else:
        if input_:
            chat_history = st.session_state.chat_history
            output = conversation_chain({'question': input_, 'chat_history': chat_history})
            # output = chain({"query": input_})
            output = output['answer']
            st.session_state.message_history.append(input_)
            st.session_state.generated.append(output)
            st.session_state.chat_history.append((input_, output))
            tts = gTTS(output)
            tts.write_to_fp(sound_file)
            st.text("Voice Over of Answer: ")

            st.audio(sound_file)
            with placeholder.container():
                message(st.session_state.message_history[-1], is_user=True, avatar_style = "adventurer", seed= "Sassy",  key="".join(set(random.choices(input_.split(), k=10)))) # display the latest message
                message(st.session_state.generated[-1], avatar_style="bottts", seed = "Milo", key="".join(set(random.choices(output.split(), k=10))))

if selected == "Question Generator":
    lottie_coding = load_lottieurl("https://lottie.host/ac6517a2-7466-4187-87f9-3ef04c04ca16/tbFBqXmQJm.json")

    with st.container():
        left, right = st.columns(2)
    with left:
        st.title(f"{selected}")
    with right:
        st_lottie(lottie_coding, height = 120, width = 160)
        
    llm2 = LLMChain(llm= ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=800, temperature = 0.2), prompt=PROMPT2)
    input_text = st.text_input("Write only the topic name from which you want questions: ")
    if input_text:
        response = llm2.run(topic=input_text)
        st.write(response)






