from llama_cpp import Llama

# Path to your Mistral model (.gguf file)
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def get_llama():
    return Llama(model_path=MODEL_PATH, n_ctx=2048)

def ask_local_llm(context, question, max_tokens=256):
    prompt = f"""[INST] You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question} [/INST]"""
    try:
        llama = get_llama()
        response = llama(prompt, max_tokens=max_tokens, temperature=0.7)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Error during LLM response: {str(e)}"

def rewrite_prompt(question):
    prompt = f"""[INST] Rewrite the following question to make it clearer and more specific:

Question: {question} [/INST]"""
    try:
        llama = get_llama()
        response = llama(prompt, max_tokens=50, temperature=0.3)
        return response["choices"][0]["text"].strip()
    except Exception:
        return question  # fallback to original

def summarize_text(text):
    prompt = f"""[INST] Summarize the following document in one sentence:

{text} [/INST]"""
    try:
        llama = get_llama()
        response = llama(prompt, max_tokens=100, temperature=0.5)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "(Failed to summarize)"
