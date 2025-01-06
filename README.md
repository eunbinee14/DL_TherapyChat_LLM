# DL_TherapyChat_LLM
### 아동 청소년 심리 상담 데이터를 기반으로 LLM을 활용한 챗봇 구현
- 4학년 2학기 딥러닝 기반 데이터 분석 아카이빙 레포지토리 입니다.

<br/><br/>
## Project Information
- 학대 피해 경험이 해가 갈수록 증가합니다.
- 아동 학대가 심하면 사망까지 이르게 됩니다.
- 실질적인 예방 대책이 여전히 부족한 실정입니다.
- 상담 챗봇을 통해 **심리적 안정**과 전문가에게 상황을 전달할 수 있는 **매개체** 역할을 기대하며 프로젝트를 진행하였습니다.
- 프로젝트 기간

    > 2024.12.02 ~ 2024.12.12

<br/><br/>
## 1. Development Environment Assign
- Google Colaboratory
- Visual Studio Code
- Python 3.10.13

<br/><br/>
## 2. Pipeline
![image](https://github.com/user-attachments/assets/0115a878-07c5-447a-a04f-7aea28c73464)

<br/><br/>
## 3. Data
> AI-Hub 아동 청소년 상담 데이터
- 2023년 상담 데이터
- 3,596건
    - train data : 86%
    - test data : 14%
 
<br/><br/>
## 4. LLM Foundation model
- GPT-2
- huggingface name : skt/kogpt2-base-v2

( 용량이 큰 파일은 업로드 하지 않았습니다. )

<br/><br/>
## 5. Fine Tuning model
> PEFT
### LoRA
- task type : CAUSAL_LM
- target modules : "c_proj"
- train batch size = 16
- gradient accumulation steps = 4
 - save steps = 500

<br/><br/>
## 6. Vector DB
pdf document
1. 심리 상담에서 상담자가 주의해야 할 내용
2. 아동 혹은 청소년 심리 평가를 수행하기 위한 질문
3. 한국어 심리 상담 데이터셋
   - 멀티턴 대화 데이터
   - 지칭 대상, 인사말, 반복어 제외
   - [Github] CounselGPT
         질의, 답변은 OpenAI API 통해 구축

> **split 188 documents into 1105 chunks**

<br/><br/>
## 7. Rag inference
- 유사도 점수 필터링 -2 ~ 2 허용
- Instruction, Context, Question, Answer

<br/><br/>
## 8. Streamlit Chat APP
webpage link : http://localhost:8501/
<img src="https://github.com/user-attachments/assets/f9da09e1-3801-41af-9293-249a4876dc44.png">



( 성능 높이기 위한 디벨롭롭 중에 있습니다. )
