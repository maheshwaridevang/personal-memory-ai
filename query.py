import chromadb
from chromadb import PersistentClient
from llm import ask_local_llm,rewrite_prompt

DB_DIR = "db"
COLLECTION_NAME = "memory"
chroma_client = PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def search_memory(query_text, top_k=3):
    results = collection.query(query_texts=[query_text], n_results=top_k)
    return results["documents"][0]

def ask_gpt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    return ask_local_llm(context, question)

if __name__ == "__main__":
    print("🧠 Personal Memory AI (Local LLM Edition)")
    while True:
        user_query = input("\n🔍 Ask something (or type 'exit'): ").strip()
        if user_query.lower() == "exit":
            break
        
        rewritten_query=rewrite_prompt(user_query)
        print(f"\n📝 Rewritten Query: {rewritten_query}")

        top_chunks = search_memory(rewritten_query)
        print("\n📄 Top Matching Chunks:\n")
        for i, res in enumerate(top_chunks):
            print(f"[{i+1}] {res.strip()}\n")

        print("🤖 Answer from Local Model:\n")
        answer = ask_gpt(top_chunks, rewritten_query)
        print(answer)
