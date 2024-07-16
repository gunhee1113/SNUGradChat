from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "huggingface/llama3-small"  # 사용할 모델 이름
model_dir = "models/llama3-small"  # 모델을 저장할 로컬 디렉토리

# 모델과 토크나이저를 다운로드하여 로컬 디렉토리에 저장
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_dir)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_dir)
