def capture_messages():
    # Connect to GHL's WebSocket endpoint
    async with GHLWebSocket() as ws:
        async for msg in ws:
            # Store in ChromaDB with metadata
            chroma_client.get_collection("User_id").upsert(
                    ids=[msg.id],
                    documents=[msg.content],
                    metadatas=[
                        {
                            "autopilot": msg.metadata.autopilot,
                            "source": "CRM",
                            "timestamp": msg.timestamp,
                            "type": "note" if not msg.metadata.autopilot else "regular",
                            "Summarize": "Summarize(msg)",
                        }
                    ]
                )
            # Forward to AI agent
            relevance_ai.process(msg)