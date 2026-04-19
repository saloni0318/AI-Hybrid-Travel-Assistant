# hybrid_chat.py (Updated for Task-03)
import json
from typing import List, Dict
from huggingface_hub import InferenceClient
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
import time
import traceback
import asyncio
from langchain.memory import ConversationBufferMemory

# -----------------------------
# Initialize clients
# -----------------------------
print("Initializing Hugging Face client for embeddings...")
hf_client = InferenceClient(model=config.HF_EMBED_MODEL, token=config.HF_TOKEN)

print("Initializing Hugging Face client for LLM...")
hf_llm_client = InferenceClient(token=config.HF_TOKEN)

print("Initializing Pinecone client...")
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Conversation memory (LangChain)
# Using ConversationBufferMemory to keep entire chat during the session.
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

# -----------------------------
# Ensure Pinecone index exists
# -----------------------------
def ensure_index():
    try:
        existing_indexes = pc.list_indexes()
        existing_indexes = [str(idx) for idx in existing_indexes]
    except Exception as e:
        print("Error listing Pinecone indexes:", e)
        existing_indexes = []

    if config.PINECONE_INDEX_NAME in existing_indexes:
        print(f"Pinecone index '{config.PINECONE_INDEX_NAME}' already exists.")
    else:
        print(f"Creating Pinecone index '{config.PINECONE_INDEX_NAME}' with dim={config.PINECONE_VECTOR_DIM}...")
        try:
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.PINECONE_VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(2)
        except Exception as e:
            if hasattr(e, "status") and e.status == 409:
                print(f"Index '{config.PINECONE_INDEX_NAME}' already exists (caught 409). Skipping creation.")
            else:
                print("Failed to create Pinecone index:", e)
                traceback.print_exc()
                raise

    return pc.Index(config.PINECONE_INDEX_NAME)

index = ensure_index()

# -----------------------------
# Connect to Neo4j
# -----------------------------
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Caching
# -----------------------------
embedding_cache: Dict[str, List[float]] = {}
query_cache: Dict[str, List[Dict]] = {}

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding using Hugging Face Inference API with caching."""
    if text in embedding_cache:
        return embedding_cache[text]

    try:
        response = hf_client.feature_extraction(text)
        if hasattr(response, "tolist"):
            vec = response.tolist()
        elif isinstance(response, list) and isinstance(response[0], list):
            vec = response[0]
        else:
            vec = response
        embedding_cache[text] = vec
        return vec
    except Exception as e:
        print(f"Error embedding text: {e}")
        return [0.0] * config.PINECONE_VECTOR_DIM

def pinecone_query(query_text: str, top_k=config.TOP_K):
    """Query Pinecone index using embedding, with caching."""
    if query_text in query_cache:
        return query_cache[query_text]

    vec = embed_text(query_text)
    try:
        res = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        matches = res.get("matches", [])
    except Exception as e:
        print("Error querying Pinecone:", e)
        matches = []

    query_cache[query_text] = matches
    print(f"DEBUG: Retrieved {len(matches)} Pinecone matches")
    return matches

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            try:
                recs = session.run(q, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r["description"] or "")[:400],
                        "labels": r["labels"]
                    })
            except Exception as e:
                print(f"Error fetching graph for node {nid}: {e}")
    print(f"DEBUG: Retrieved {len(facts)} graph facts")
    return facts

# -----------------------------
# Search Summary
# -----------------------------
def search_summary(pinecone_matches, graph_facts, top_n=3) -> str:
    """Generate a concise summary of top nodes from vector DB and graph without duplicates."""
    # Vector nodes
    vec_seen = set()
    vec_top = []
    for m in pinecone_matches:
        name = m['metadata'].get('name', 'Unknown')
        if name not in vec_seen:
            vec_seen.add(name)
            vec_top.append(name)
        if len(vec_top) >= top_n:
            break

    # Graph nodes
    graph_seen = set()
    graph_top = []
    for f in graph_facts:
        name = f['target_name']
        if name not in graph_seen:
            graph_seen.add(name)
            graph_top.append(name)
        if len(graph_top) >= top_n:
            break

    return f"Top vector nodes: {', '.join(vec_top)}.\nTop graph nodes: {', '.join(graph_top)}."

# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(user_query, pinecone_matches, graph_facts, memory_text: str = ""):
    """Build a prompt combining vector DB matches, graph facts, and summary."""
    system = (
        "You are an expert travel assistant with knowledge of locations, attractions, "
        "activities, and local insights. Your goal is to provide clear, accurate, "
        "and engaging answers to user queries using the provided semantic search results "
        "and graph facts. Always cite cities, attractions, or activities by their "
        "name followed by their node ID in parentheses like this: (activity_150). Avoid repeating the same information. "
        "Use the context from vector database, graph facts, and the top-node summary to enrich your response. "
        "Be concise, informative, and user-friendly. "
        "If the user's query is about travel planning, suggestions, or cultural insights, "
        "organize your response in a readable structure with bullet points or numbered steps where appropriate. "
        "For any general questions, provide explanations, tips, or comparisons as needed, "
        "still citing relevant node IDs when referencing specific places."
    )

    # Vector & graph context formatting
    vec_context = [
        f"- {m.get('metadata', {}).get('name','Unknown')} (id: {m['id']}), type: {m.get('metadata',{}).get('type','')}, score: {m.get('score')}"
        for m in pinecone_matches
    ]
    graph_context = [
        f"- {f['target_name']} (id: {f['target_id']}) connected to {f['source']} via {f['rel']}: {f['target_desc']}"
        for f in graph_facts
    ]

    # Get summary and prepend to prompt
    summary_text = search_summary(pinecone_matches, graph_facts)

    # If memory_text provided, show recent conversation first
    memory_section = ""
    if memory_text:
        memory_section = f"Conversation history:\n{memory_text}\n\n"

    prompt_text = (
        f"{system}\n\n"
        f"{summary_text}"
        f"{memory_section}"
        f"User query: {user_query}\n\n"
        "Top semantic matches (vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
        "Graph facts (relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
        "Based on the above, answer the user's question. "
        "Provide 2–3 concrete itinerary steps or tips if applicable, citing node IDs."
    )

    return prompt_text

# -----------------------------
# Call LLM
# -----------------------------
def call_chat(prompt_text: str):
    """Call Hugging Face chat LLM and automatically continue if output seems truncated."""
    full_response = ""
    while True:
        try:
            response = hf_llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=config.MAX_TOKENS,
                temperature=0.7,
            )

            if hasattr(response, "choices") and len(response.choices) > 0:
                chunk = response.choices[0].message["content"]
            else:
                chunk = "[No content returned from model]"

            full_response += chunk

            # Heuristic: check if response ends with a likely cutoff
            if chunk.strip().endswith((".", "!", "?", "\n")):
                break  # natural end
            else:
                # Ask model to continue from last text
                prompt_text = f"Continue the previous answer:\n{full_response}"
        except Exception as e:
            print("Error calling LLM:", e)
            if not full_response:
                return "[Error generating response]"
            break

    return full_response


# -----------------------------
# Async helpers
# -----------------------------
async def async_embed_text(text: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_text, text)

async def async_fetch_graph(node_ids: List[str]):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_graph_context, node_ids)

async def process_query_async(query: str):
    matches = pinecone_query(query)
    match_ids = [m["id"] for m in matches]
    embed_task = async_embed_text(query)
    graph_task = async_fetch_graph(match_ids)
    embeddings, graph_facts = await asyncio.gather(embed_task, graph_task)
    return matches, graph_facts

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit","quit"):
            break

        # Async query processing
        matches, graph_facts = asyncio.run(process_query_async(query))

        # Load conversation memory (string). If empty, load_memory_variables returns "".
        mem_vars = memory.load_memory_variables({})
        mem_text = mem_vars.get("chat_history", "")  # string transcript

        summary_text = search_summary(matches, graph_facts)
        print(f"\n=== Summary ===\n{summary_text}\n")

        # Build prompt including memory_text (keeps behavior if mem_text == "")
        prompt_text = build_prompt(query, matches, graph_facts, memory_text=mem_text)
        answer = call_chat(prompt_text)

        # Save the turn to memory so it helps future turns
        try:
            memory.save_context({"input": query}, {"output": answer})
        except Exception as e:
            # do not crash on memory save; just log
            print("Warning: failed to save to memory:", e)

        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    interactive_chat()
