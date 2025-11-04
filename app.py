from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_community.llms import Ollama

# -------------------------
# Flask App Initialization
# -------------------------
app = Flask(__name__)
load_dotenv()

# -------------------------
# Environment Variables
# -------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -------------------------
# Embeddings and Pinecone Setup
# -------------------------
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -------------------------
# LLM (Ollama Model)
# -------------------------
chatModel = Ollama(model="phi3")

# -------------------------
# Prompt and Chain Setup
# -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -------------------------
# Predefined Small Talk Replies
# -------------------------
BASIC_RESPONSES = {
    "hi": "Hey there! 👋 How can I help you today?",
    "hello": "Hello! 😊 How are you doing?",
    "hey": "Hey! What’s up?",
    "how are you": "I’m great, thanks for asking! How about you?",
    "bye": "Goodbye! 👋 Take care and stay healthy!",
    "goodbye": "See you soon! Stay safe!",
    "thanks": "You're very welcome! 😊",
    "thank you": "No problem! Happy to help 😊",
    "who are you": "I’m your friendly medical assistant bot 🤖 here to answer health questions!",
    "what can you do": "I can answer questions about diseases, symptoms, and medical concepts in simple terms.",
    "help": "Sure! Ask me about any medical condition, symptom, or health topic — I’ll try my best to explain."
}

# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"].strip().lower()
    print(f"User: {msg}")

    # Step 1: Check for small talk first
    if msg in BASIC_RESPONSES:
        bot_reply = BASIC_RESPONSES[msg]
        print("Bot (Rule-based):", bot_reply)
        return bot_reply

    # Step 2: Otherwise, use the RAG + Ollama pipeline
    try:
        response = rag_chain.invoke({"input": msg})
        bot_reply = response["answer"]
        print("Bot (Model):", bot_reply)
        return bot_reply

    except Exception as e:
        print("Error:", str(e))
        return "Sorry, something went wrong while processing your request."


# -------------------------
# Run the Flask App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
