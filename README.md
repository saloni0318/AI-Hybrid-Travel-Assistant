# AI-Hybrid-Travel-Assistant
Project Workflow: AI-Powered Hybrid Travel Assistant

This section describes how the system operates, broken down into clear and simple steps.

Step-by-Step Execution
Start the application (hybrid_chat.py)
The program is launched, initializes required settings, and asks the user to input a travel-related query.
Import settings from config.py
Necessary configurations such as API credentials, model specifications, and index information are loaded.
Create query embeddings using a Hugging Face model
The user’s input is transformed into a high-dimensional vector representation (768 dimensions) to enable semantic understanding.
Perform similarity search with Pinecone
The generated vector is compared against stored embeddings in the Pinecone database to identify closely related results.
Retrieve connected data from Neo4j graph database
Based on the matched results, related entities such as destinations, activities, and their relationships are fetched.
Integrate data for hybrid reasoning
The system combines multiple inputs:
The original user query
Results from Pinecone similarity search
Relationship data from Neo4j
Conversation history stored via LangChain
Generate a response using an LLM (Hugging Face / OpenAI)
A structured prompt is created using the combined data and passed to the language model to produce an intelligent reply.
Display the generated output
The system presents the final result (e.g., travel suggestions, itineraries) to the user in the terminal.
Maintain conversation context with LangChain Memory
The interaction is saved so that future queries can benefit from previous context.
Process subsequent queries
For follow-up questions, the system uses stored memory to provide more personalized and context-aware responses.
Overall Data Pipeline

User Input → Embedding Generation (Hugging Face) → Vector Search (Pinecone) → Graph Retrieval (Neo4j) → Data Integration → LLM Processing → Response Output → Memory Storage

Overview

This system combines multiple technologies to deliver a smarter travel assistant:

Pinecone enables semantic similarity search
Neo4j provides relationship-based insights
LLMs (Hugging Face / OpenAI) handle natural language understanding and generation
LangChain Memory ensures context retention across conversations

Together, these components create a responsive and context-aware travel assistant capable of handling complex user queries.
