import time
import logging
from typing import List, Dict

# OpenTelemetry pour le tracing
from opentelemetry.trace import get_tracer

# Base de données vectorielle
import chromadb

# Transformers et modèles de langage
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Anonymisation PII
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Métriques (à adapter selon l'implémentation réelle)
from prometheus_client import Counter

# Initialisation des composants (à ajouter dans le code)
tracer = get_tracer(__name__)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Modèle d'embedding
mistral7b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1") 
latency_counter = Counter('latency_ms', 'Processing latency')


# Constants
BATCH_SIZE = 32
NOTE_TYPE = "crm_note"  # From CRM settings
REGULAR_TYPE = "crm_message"
MAX_HISTORY_DAYS = 365

def redact_pii(text: str) -> str:
    """Redact sensitive information from text"""
    analysis = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=analysis).text

def detect_industry(messages: list[str]) -> str:
    """Dynamically detect industry using LLM"""
    industry_prompt = f"""Analyze these messages to determine industry:
    {" ".join(messages[:3])}
    
    Options: tech, healthcare, finance, retail, other
    Respond only with the industry name."""
    
    return mistral7b.generate(
        prompt=industry_prompt,
        max_tokens=10
    ).strip().lower()

def handle_new_conversation(lead_id: str, query: str) -> dict:
    with tracer.start_as_current_span("new_conversation"):
        start_time = time.perf_counter()
        
        try:
            # 1. Retrieve and redact history
            history = chroma.query(
                where={"lead_id": lead_id},
                include=["documents", "metadatas"],
                limit=MAX_HISTORY_DAYS
            )
            
            # Redact PII in batch
            redacted_docs = [redact_pii(doc) for doc in history["documents"]]
            
            # 2. Batch embed documents
            embeddings = model.encode(
                redacted_docs,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True
            )
            
            # 3. Process metadata
            processed_messages = []
            for doc, meta, emb in zip(redacted_docs, history["metadatas"], embeddings):
                msg = {
                    "content": doc,
                    "metadata": meta,
                    "embedding": emb,
                    "days_old": (time.time() - meta["timestamp"]) / 86400
                }
                processed_messages.append(msg)
            
            # 4. Hybrid scoring (recency + relevance)
            query_embedding = model.encode(query, convert_to_tensor=True)
            current_time = time.time()
            
            for msg in processed_messages:
                # Time decay: 0.7^t where t is days
                time_weight = 0.7 ** (msg["days_old"] / 30)  # Monthly decay
                similarity = torch.nn.functional.cosine_similarity(
                    msg["embedding"], query_embedding, dim=0
                )
                msg["score"] = (0.6 * similarity) + (0.4 * time_weight)
            
            # 5. Select top messages
            sorted_messages = sorted(
                processed_messages,
                key=lambda x: x["score"],
                reverse=True
            )[:5]
            
            # 6. Dynamic industry detection
            industry = detect_industry([msg["content"] for msg in sorted_messages])
            
            # 7. Generate CoT context
            cot_prompt = f"""**Industry**: {industry}
            **Messages**:
            {"".join([f"- {msg['content']}\n" for msg in sorted_messages])}
            
            Analyze using:
            1. Key business needs
            2. Urgent requirements
            3. Historical patterns"""
            
            context = mistral7b.generate(
                prompt=cot_prompt,
                max_tokens=200
            )
            
            # 8. Initialize agent
            agent = relevance_ai.initialize_agent(
                context=context,
                industry_template=industry
            )
            
            # Log performance
            latency = (time.perf_counter() - start_time) * 1000
            latency_counter.add(latency, {"type": "new_conversation"})
            
            return {
                "agent": agent,
                "latency_ms": round(latency, 2),
                "messages_processed": len(processed_messages)
            }
            
        except Exception as e:
            logger.error(f"Conversation init failed: {str(e)}")
            # Fallback to rules-based agent
            return relevance_ai.get_fallback_agent(lead_id)