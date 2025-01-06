import streamlit as st
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

CHROMA_PATH = "./chroma"  # 업로드된 Chroma 벡터 DB 파일 경로
# 모델 및 벡터 DB 경로 설정
base_model_path = r"C:\Users\bhks0\Desktop\LLM 응용1\models\Foundation_model"  # LLM 모델 경로
peft_adapter_path = r"C:\Users\bhks0\Desktop\LLM 응용1\models\Tuning_model"  # PEFT 어댑터 경로

# LLM 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
base_model.resize_token_embeddings(len(tokenizer))  # tokenizer에서 사용하는 어휘 크기
peft_config = PeftConfig.from_pretrained(peft_adapter_path, local_files_only=True)
model = PeftModel.from_pretrained(base_model, peft_adapter_path, local_files_only=True)

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

modelPath = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # open source embedding model 

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device': 'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': True}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

# Streamlit 앱 설정
st.set_page_config(page_title="Chatbot with RAG", page_icon="🤖")
st.title("동행")
st.write("**🌳계속 함께 있을게요🌳**")

# 중복 제거
def clean_response(text):
    """Remove duplicate words, sentences, and excessive repetition."""
    sentences = text.split(". ")  # 문장 단위로 분리
    unique_sentences = list(dict.fromkeys(sentences))  # 중복 제거
    cleaned_text = ". ".join(unique_sentences).strip()

    # 특정 단어나 문장의 반복 제거
    excessive_patterns = re.findall(r"(\b\w+\b)( \1)+", cleaned_text)  # 반복 단어 감지
    for pattern in excessive_patterns:
        cleaned_text = re.sub(rf"{pattern[0]}( {pattern[0]})+", pattern[0], cleaned_text)

    return cleaned_text

# RAG 검색 및 모델 추론 함수
def query_rag(query_text):
    # 벡터 DB에서 검색하여 문서 반환
    embedding_function = embeddings

    # Load the existing Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_relevance_scores(query_text, k=2)
    
    # 유사도 점수 필터링
    valid_results = [res for res in results if -2 <= res[1] <= 2]
    if not valid_results:
        return "관련된 응답을 찾을 수 없습니다."
    
    # 가장 높은 유사도 점수 결과 선택
    best_result = max(valid_results, key=lambda x: x[1])
    return best_result[0].page_content


def model_run(context_text, query_text, prompt_template):
    # 검색된 문서에서 컨텍스트 추출
    context_text = " ".join([context_text])

    # Prompt 템플릿 생성
    prompt_template = PromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # pad_token_id를 eos_token_id로 설정
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    
    # 모델에 입력하고 응답 생성
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # 모델 추론 (생성 결과)
    with torch.no_grad():
      outputs = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs['attention_mask'], 
        pad_token_id=tokenizer.pad_token_id,
        max_length=256, num_return_sequences=1,
        no_repeat_ngram_size=3,  # 3그램 반복 방지
        repetition_penalty=2.0,   # 반복 생성 억제
        temperature=0.7  #샘플링의 다양성 설정
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(response_text)


def generate_response(input_text):
  query_text = input_text
  prompt_template = """
    Context: {context} \n
    Question: {question} \n
  """
  # Let's call our function we have defined
  rag_text = query_rag(query_text)
  response_text = model_run(rag_text, query_text, prompt_template)
  
  # and finally, inspect our final response!
  st.info(response_text)

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        " ",
    )
    submitted = st.form_submit_button("Submit")
    if submitted :
        generate_response(text)

