# 🌍 AI Hybrid Travel Assistant

An intelligent travel assistant that combines vector search, graph databases, and large language models to deliver personalized and context-aware travel recommendations.

---

## 🚀 Overview

The **AI Hybrid Travel Assistant** is built to provide smart travel guidance by integrating multiple AI techniques. Instead of relying on a single approach, it combines semantic understanding, relationship-based reasoning, and conversational memory to generate accurate and meaningful responses.

This hybrid design allows the system to understand user intent better and provide more relevant travel suggestions.

---

## 🧠 Features

* 🔍 Semantic search using vector embeddings
* 🧩 Graph-based reasoning using relationships
* 💬 Context-aware multi-turn conversations
* 🗺️ Personalized travel recommendations
* ⚡ Efficient and scalable system design

---

## 🏗️ System Architecture

```
User Query
   ↓
Embedding Model (Hugging Face)
   ↓
Pinecone Vector Search
   ↓
Neo4j Graph Query
   ↓
Hybrid Data Fusion
   ↓
LLM (Hugging Face / OpenAI)
   ↓
Response Generation
   ↓
LangChain Memory Storage
```

---

## ⚙️ Tech Stack

* **Language**: Python
* **Embeddings**: Hugging Face Transformers
* **Vector Database**: Pinecone
* **Graph Database**: Neo4j
* **LLM**: Hugging Face / OpenAI
* **Framework**: LangChain

---

## 📂 Project Structure

```
├── hybrid_chat.py       # Entry point of the application
├── config.py            # Stores API keys and configurations
├── embeddings/          # Handles embedding generation
├── vector_db/           # Pinecone integration
├── graph_db/            # Neo4j queries and logic
├── memory/              # Conversation memory (LangChain)
└── README.md            # Project documentation
```

---

## 🔄 Workflow

1. Run `hybrid_chat.py` to start the application
2. Load configuration settings from `config.py`
3. Convert user query into embeddings
4. Perform similarity search in Pinecone
5. Retrieve related data from Neo4j
6. Combine all data for hybrid reasoning
7. Send structured prompt to LLM
8. Generate and display response
9. Store conversation using memory
10. Use stored context for follow-up queries

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-hybrid-travel-assistant.git
cd ai-hybrid-travel-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory and add:

```
OPENAI_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key
NEO4J_URI=your_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

---

## ▶️ Usage

Run the application using:

```bash
python hybrid_chat.py
```

Example query:

```
Suggest a 5-day trip plan to vietnam with nature and adventure activities
```

---

## 📌 Use Cases

* Travel itinerary generation
* Destination discovery
* Activity recommendations
* Conversational travel assistance

---

## 🔮 Future Enhancements

* Integration with real-time travel APIs
* Voice-based assistant support
* Web or mobile interface
* Personalized recommendations using user profiles



