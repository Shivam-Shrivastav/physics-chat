


import streamlit as st
from streamlit_chat import message
import os
import random
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie



EXAMPLE_NO = 1



st.set_page_config(page_title="Physics Chatbot ",
                       page_icon=":books:")
def streamlit_menu(example=1):
    if example == 1:
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Chat (QNA)"],  # required
                icons=["chat", "question", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

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

        Physics Chatbot
        
        """)
    with right:
        st_lottie(lottie_coding, height = 120)

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



    placeholder = st.empty() # placeholder for latest message
    input_ = get_text()
    if len(st.session_state.message_history) > 0 and input_ == st.session_state.message_history[-1]:
        pass
    else:
        if input_:
            chat_history = st.session_state.chat_history
            # output = Gouravs_Model(input)
            # output = chain({"query": input_})
            # output = output['answer']
            st.session_state.message_history.append(input_)
            st.session_state.generated.append(output)
            st.session_state.chat_history.append((input_, output))
            with placeholder.container():
                message(st.session_state.message_history[-1], is_user=True, avatar_style = "adventurer", seed= "Sassy",  key="".join(set(random.choices(input_.split(), k=10)))) # display the latest message
                message(st.session_state.generated[-1], avatar_style="bottts", seed = "Milo", key="".join(set(random.choices(output.split(), k=10))))






