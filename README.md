🤖 AI Meeting Summarizer & Chat Assistant

A full-featured system for summarizing meeting transcripts, storing structured metadata, and enabling intelligent retrieval-based Q&A.

🚀 Overview

This project is a complete meeting intelligence platform built with OpenAI, DeepSeek, LangChain, ChromaDB, and Gradio.
It allows you to:

Generate structured meeting summaries in Markdown

Translate summaries into multiple languages

Store summaries in a vector database with rich metadata

Visualize vector embeddings (t-SNE)

Ask questions about meetings with retrieval-augmented generation

Interact via a modern Gradio UI with streaming responses

Automatically save summaries into an organized knowledge base

✨ Key Features
📝 Automated Meeting Summarization

Supports GPT-4o, GPT-4o-mini, and DeepSeek models

Outputs consistent, structured Markdown:

Title

Date

Attendees

Discussion points

Decisions made

Action items

🌍 Multilingual Translation

Translates summaries while preserving Markdown formatting

Supports English, Chinese, Japanese, Korean, French, German, Spanish

📁 Intelligent Knowledge Base

Saves each summary using its meeting date

Extracts metadata:

Date

Attendees

Discussion points

Decisions

Action items

Titles

Stores everything inside ChromaDB with OpenAI embeddings

🔍 Smart Meeting Q&A (RAG)

Uses LangChain’s ConversationalRetrievalChain

MMR search improves diversity & relevance

Ensures answers are strictly meeting-related

Provides source citations

📊 Visual Dashboard

Automatic bar chart of summaries per month

t-SNE visualization of vector embeddings

All rendered inside Gradio UI

🖥️ Modern Gradio Interface

Multiple tabs (Summarize, Chat)

Real-time streaming updates

Custom theme

Example meeting notes for quick testing

🧠 Architecture Overview