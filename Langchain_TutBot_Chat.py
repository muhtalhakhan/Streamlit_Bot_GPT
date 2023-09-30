"""Python file to serve as the frontend"""
import streamlit as st
import os
from streamlit_chat import message

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from openai.error import RateLimitError
import backoff

file_path_lawyerbot = "LawyerBot.txt"
file_path_educatorbot = "educator_bot.txt"

with open(file_path_lawyerbot, "r") as file:
    lawyerbot = file.read()

with open(file_path_educatorbot, "r") as file:
    educatorbot = file.read()

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff(**kwargs):
    response = OpenAI.Completion.create(**kwargs)
    return response

st.set_page_config(page_title="Tutor Bot", page_icon=":robot:")

# From here down is all the Streamlit UI.
st.header("Bot - GPT")

os.environ["OPENAI_API_KEY"] = st.text_input(key='OpenAI_Key', label="Enter Your Key", value="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", type="password")

def load_chain(selected_option):
    from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

    system_templates = {
        "Lawyer Bot": lawyerbot,
        "Educator Bot": educatorbot
    }

    selected_template = system_templates[selected_option]

    messages = [
        SystemMessagePromptTemplate.from_template(selected_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256, openai_api_key=os.environ["OPENAI_API_KEY"])  # Modify model_name if you have access to GPT-4

    chain = LLMChain(llm=llm, verbose=True, **chain_type_kwargs)

    return chain

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def execute_query(query):
    output = chain(query)
    return output

chain = None

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty()

form = st.form("my_form")

options = ["Lawyer Bot", "Educator Bot"]
selected_option = form.selectbox("Select an option", options)

if selected_option:
    chain = load_chain(selected_option)

user_input = form.text_input("You:", "Hi, How can I learn from you?", key="input")
submitted_flag = form.form_submit_button()

if submitted_flag:
    with placeholder.container():
        if user_input:
            output = execute_query(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output.get('text'))

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))
