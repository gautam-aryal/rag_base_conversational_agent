from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Generator:
  def __init__(self):
    pass

  def chat_with_llama(self, prompt):
    llm = OllamaLLM(model="llama3:8b-instruct-q8_0")
    response = llm.invoke(prompt)
    return response


  def chat_with_gpt_oss_20b(self, prompt):
    llm = OllamaLLM(model="gpt-oss:20b")
    response = llm.invoke(prompt)
    return response

  def chat_with_mistral_15b(self, prompt):
    llm = OllamaLLM(model="mistral:instruct")
    response = llm.invoke(prompt)
    return response

  def chat_with_mistral_7b(self, prompt):
    llm =  OllamaLLM(model="mistral:7b")
    response = llm.invoke(prompt)
    return response

  def chat_with_gimma3_12b(self, prompt):
    llm = OllamaLLM(model="gemma3:12b-it-q8_0")
    response = llm.invoke(prompt)
    return response

  def chat_with_qwen3_8b(self, prompt):
    llm = OllamaLLM(model="qwen3:8b")
    response = llm.invoke(prompt)
    return response

  def chat_with_deepseek_r1_8b(self, prompt):
    llm = OllamaLLM(model="deepseek-r1:8b")
    response = llm.invoke(prompt)
    return response

  def chat_with_llama3_3b(self, prompt):
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    rag_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = rag_pipeline(prompt, max_new_tokens=1000, do_sample=True)[0]['generated_text']
    return response.split("Answer:")[-1].strip()