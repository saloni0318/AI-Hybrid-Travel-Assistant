# 🌍 AI Hybrid Travel Assistant

An AI-powered travel assistant that combines **vector search (Pinecone)**, **graph databases (Neo4j)**, and **large language models** to generate intelligent, context-aware travel recommendations.

---

## 🚀 Overview

This project implements a **hybrid AI system** that enhances travel planning by combining:

* Semantic understanding using embeddings
* Relationship-based insights using graph databases
* Natural language responses using LLMs
* Context retention using conversational memory

The system is capable of answering complex travel queries and supporting multi-turn conversations.

---

## 🧠 Key Features

* 🔍 Semantic search with embeddings (Hugging Face)
* 🧩 Graph-based reasoning using Neo4j
* 💬 Context-aware conversations with LangChain memory
* 🗺️ Intelligent itinerary and travel suggestions
* ⚡ Hybrid architecture for better accuracy

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
LLM Response Generation
   ↓
Memory Storage (LangChain)
```

---

## 📂 Project Structure

```
├── README.md                   # Project documentation
├── architecture_diagram.png    # System architecture diagram
├── config.py                   # API keys and configuration settings
├── hybrid_chat.py              # Main application (entry point)
├── improvements.md             # Future improvements and ideas

├── load_to_neo4j.py            # Script to load dataset into Neo4j
├── visualize_graph.py          # Graph visualization script
├── neo4j_viz.html              # Exported graph visualization

├── pinecone_upload.py          # Upload embeddings to Pinecone
├── vietnam_travel_dataset.json # Travel dataset

├── requirements.txt            # Dependencies
```

---

## ⚙️ Tech Stack

* **Programming Language**: Python
* **Embeddings**: Hugging Face Transformers
* **Vector Database**: Pinecone
* **Graph Database**: Neo4j
* **LLM**: OpenAI / Hugging Face
* **Framework**: LangChain

---

## 🔄 Workflow

1. Run `hybrid_chat.py`
2. Load configuration from `config.py`
3. Convert user query into embeddings
4. Perform similarity search in Pinecone
5. Fetch related entities from Neo4j
6. Combine semantic + relational data
7. Send structured prompt to LLM
8. Generate and display response
9. Store conversation in memory
10. Handle follow-up queries using context

---

## 🛠️ Setup Instructions

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

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key
NEO4J_URI=your_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

---

## ▶️ How to Run

### Step 1: Upload data to Pinecone

```bash
python pinecone_upload.py
```

### Step 2: Load data into Neo4j

```bash
python load_to_neo4j.py
```

### Step 3: Run the chatbot

```bash
python hybrid_chat.py
```

---

## 📊 Visualization

* Graph visualization: `neo4j_viz.html`
* Architecture diagram: `architecture_diagram.png`

---

## 📌 Example Query

```
Plan a 4-day trip to Vietnam with cultural and adventure activities
```

---

## 🔮 Future Improvements

* Real-time travel API integration
* Voice-based interaction
* Web or mobile interface
* Personalized recommendations using user profiles

