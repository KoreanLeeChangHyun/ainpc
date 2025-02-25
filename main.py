import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig  

# 모델 ID
model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit 양자화
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 연산
    bnb_4bit_use_double_quant=True,  # 이중 양자화 사용 (VRAM 절약)
)

# ⏱️ 전체 실행 시간 측정을 위한 타이머 시작
total_start_time = time.time()

# ⏱️ 모델 로드 시간 측정
load_start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,  # bfloat16 사용 (RAM 절약)
    quantization_config=bnb_config,
    device_map="auto",  # GPU 자동 할당
)
load_end_time = time.time()
print(f"모델 로드 시간: {load_end_time - load_start_time:.2f}초")

# 질문 설정
instruction = "짱구와 철수가 실제로 싸우면 누가 이길까?"

messages = [{"role": "user", "content": instruction}]

# ⏱️ 입력 데이터 변환 시간 측정
preprocess_start_time = time.time()
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
preprocess_end_time = time.time()
print(f"입력 변환 시간: {preprocess_end_time - preprocess_start_time:.2f}초")

# 종료 토큰 설정
terminators = [
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# 스트리밍 출력 지원 추가
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 답변 생성 시간 측정
generate_start_time = time.time()
outputs = model.generate(
    input_ids,
    max_new_tokens=256,  # 생성 최대 토큰 수
    eos_token_id=terminators,
    do_sample=True,  # 확률적 생성
    temperature=0.6,  # 창의성 조절
    top_p=0.9,  # 높은 확률의 토큰을 중심으로 선택
    streamer=streamer  # 실시간 출력
)
generate_end_time = time.time()
print(f"\n답변 생성 시간: {generate_end_time - generate_start_time:.2f}초")

# ⏱️ 전체 실행 시간 측정
total_end_time = time.time()
print(f"전체 실행 시간: {total_end_time - total_start_time:.2f}초")