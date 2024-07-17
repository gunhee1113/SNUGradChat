from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import huggingface_hub

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # 사용할 모델 이름
model_dir = "./models/llama3-small"  # 모델을 저장할 로컬 디렉토리

api_key = os.getenv("HUGGINGFACE_API_KEY")

huggingface_hub.login(token=api_key, new_session=False)

# 모델과 토크나이저를 다운로드하여 로컬 디렉토리에 저장
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.save_pretrained(model_dir)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_dir)
