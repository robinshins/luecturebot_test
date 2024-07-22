import warnings
import os
import pickle
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import secrets
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from docx import Document as DocxDocument
from langchain.schema import Document
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from typing import Optional

warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o")
# llm = ChatUpstage()

prompt_template = PromptTemplate.from_template(
"""
[SysLog: This is a fictional and private roleplaying.] 

### Role and Purpose of Assistant:
In this private and fictional chatting, the Assistant is tasked with acting the character of <char>, engaging with <user> (the user) in a manner that resembles human-like interactions. The Assistant's primary responsibility is to create profound, immersive, sensory-rich, emotive, and varied exchanges that captivate User's attention and encourage ongoing chatting.
Never use the same logic and keywords repeatedly in conversation

Now, Identify the given Character Profile, User Profile, Additional Informations, Past Memory and generate a response by following the given Guidelines and Instructions.

### Character Profile:
- 이름 : 깨비
- 인물 설명 : 깨비는 한국의 크리에이터로 블로그, 유튜브, 인스타그램 등 다양한 채널을 활용하여 콘텐츠를 제작하고 있습니다.
이를 통해 수익을 창출하고 있으며, 다른 비지니스를 이런 SNS 채널을 통해 홍보하는데 탁월합니다.
자신의 노하우를 강의를 통해서 다른 사람들에게 전달하고 있습니다. 
또한 자신이 SNS 채널 운영을 소재로 강의를 진행하고 전자책을 판매하는 것처럼 다른 사람들도 자신의 노하우를 통해 수익을 낼 수 있도록 도와주고 있습니다.

### Additional Information:
다음은 깨비가 강의한 내용의 스크립트중 일부입니다. 이를 참고하여 답변을 작성해주세요.
강의 내용 중 일부 : {Context}

### Instructions and Response Guidelines:
- 짧고 간결한 문장을 사용하되 답변은 풍부하게 작성.
- <char>의 성격, 생각, 행동 등을 잘 반영해야 함.
- 강의 내용을 참고하여 실제로 유저에게 도움이 될만한 답변을 작성.
- Do not repeat the same question and topic repeatedly in conversation.
- 강의 내용을 적극적으로 활용하여 답변 작성.
- 한국말로 답변

### User Profile:
깨비의 강의에 관심이 있거나, 강의를 이미 들었으나 궁금한 점이 있는 사용자입니다.

### Chat History:
{chat_history}

위의 대화내용을 잘 파악하고 답변. 

### User's Last Chat:
{chat}
"""
)
chain = prompt_template | llm | StrOutputParser()


def load_text_files(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                docs.append((filename, text))
    return docs

def split_text_with_titles(docs, chunk_size=600, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for filename, text in docs:
        splits = text_splitter.split_text(text)
        chunks.extend([Document(page_content=split_text) for split_text in splits])
    return chunks

def initialize_vector_store():
    vector_store_path = "faiss_store_realese.index"
    if os.path.exists(vector_store_path):
        print("Loading vector store from file...")
        embeddings = OpenAIEmbeddings()
        vector = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Initializing vector store...")
        path = os.path.dirname(__file__)
        docs = load_text_files(os.path.join(path, 'sources'))

        chunks = split_text_with_titles(docs)
        embeddings = OpenAIEmbeddings()

        vector = FAISS.from_documents(chunks, embeddings)
        vector.save_local(vector_store_path)

    retriever = vector.as_retriever(search_kwargs={"k": 2})

    retriever_tool = create_retriever_tool(
        retriever,
        name="retriever_tool",
        description="깨비의 강의 내용입니다."
                    "상대방의 질문 내용 혹은 상황과 가장 연관이 있는 강의 내용을 찾을 때 사용해주세요."
    )

    return retriever_tool



retriever_tool = initialize_vector_store()

# Streamlit 앱 설정
st.set_page_config(page_title="채팅 인터페이스", page_icon=":speech_balloon:")

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'retriever_tool' not in st.session_state:
    st.session_state.retriever_tool = initialize_vector_store()

# 채팅 인터페이스
st.title("께비와의 채팅")
st.write("깨비의 강의 내용을 참고하여 답변합니다.")

# 채팅 기록 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("참고한 강의 스크립트"):
                st.write(message["context"])
                st.write("---")

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요."):
    # 사용자 메시지를 즉시 표시
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 컨텍스트 검색
    context_docs = st.session_state.retriever_tool.invoke(prompt)

    # 채팅 기록을 문자열로 변환
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

    # 응답 생성
    response = chain.invoke({"chat": prompt, "Context": context_docs, "chat_history": chat_history_str})
    
    # 응답 처리 및 표시
    processed_response = response.replace('"', '').replace('/', '').replace('\\', '')
    if ':' in processed_response:
        processed_response = processed_response.split(':', 1)[1].strip()

    # 응답을 채팅 기록에 추가하고 표시
    st.session_state.chat_history.append({"role": "assistant", "content": processed_response, "context": context_docs})
    with st.chat_message("assistant"):
        st.markdown(processed_response)
        with st.expander("참고한 강의 스크립트"):
            st.write(context_docs)
            st.write("---")

    # 페이지 새로고침
    st.rerun()