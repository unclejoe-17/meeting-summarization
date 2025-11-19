# 🧠 AI Meeting Summarizer & Retrieval System

A full-featured Meeting Summarization, Translation, Vector Search, and
Conversational Retrieval system built with **Python**, **LangChain**,
**OpenAI**, **DeepSeek**, **ChromaDB**, and **Gradio**.

This document explains the structure, purpose, and functionality of the
source code so you can include it as your project README on GitHub.

------------------------------------------------------------------------

## 📌 Overview

This project provides an interactive dashboard and chatbot interface to:

-   Summarize meeting transcripts using GPT or DeepSeek\
-   Translate summaries into multiple languages\
-   Save structured meeting summaries in Markdown format\
-   Build a vector database of meeting notes using ChromaDB\
-   Query meeting history conversationally\
-   Visualize embedding clusters with t-SNE\
-   Provide a fully interactive web UI using Gradio

------------------------------------------------------------------------

## 📂 Project Structure

    knowledge-base/
       └── meetings/         # Auto‑saved meeting summaries
    vector_db/               # Persistent Chroma vector database
    main.py                  # Main application (your provided code)
    .env                     # API keys for OpenAI & DeepSeek

------------------------------------------------------------------------

## ⚙️ Key Components

### 1. **Environment Setup**

Loads API keys using `dotenv` and initializes both OpenAI and DeepSeek
clients:

``` python
load_dotenv(override=True)
openai = OpenAI()
deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
```

------------------------------------------------------------------------

### 2. **Meeting Summarization**

The system enforces a strict output format using a custom System Prompt:

-   Title\
-   Date\
-   Human attendees\
-   Discussion points\
-   Decisions\
-   Action items\
-   Markdown‑only output

Summaries stream live using either GPT‑4o or DeepSeek.

------------------------------------------------------------------------

### 3. **Saving Summaries**

Summaries are automatically saved as:

    YYYY-MM-DD_title_fragment.md

Files are stored under:

    knowledge-base/meetings/

------------------------------------------------------------------------

### 4. **Translation System**

The summarizer also supports translation into multiple languages while
keeping the markdown structure intact.

------------------------------------------------------------------------

### 5. **Metadata Extraction**

Each meeting summary is parsed to extract searchable metadata:

-   Date & Month\
-   Attendees & Count\
-   Discussion Points\
-   Decisions\
-   Action Items\
-   Title

This metadata enriches vector search relevance.

------------------------------------------------------------------------

### 6. **Vector Database (ChromaDB)**

All meeting summaries are embedded using OpenAI embeddings and stored in
a persistent ChromaDB instance.

Search uses **MMR (Maximal Marginal Relevance)** for diverse, relevant
context retrieval.

------------------------------------------------------------------------

### 7. **Visualization**

Embeddings are reduced via t‑SNE and plotted using Plotly to visualize
clusters of meetings.

------------------------------------------------------------------------

### 8. **Conversational Meeting Assistant**

A chat interface allows users to query across historical meetings:

-   "What were the decisions in October?"\
-   "Show meetings involving the Marketing team."\
-   "What are all action items assigned to Alex?"

All answers include source citations.

------------------------------------------------------------------------

### 9. **Gradio UI**

A polished Gradio dashboard includes:

-   Meeting summary generator\
-   Output editor\
-   Translation module\
-   Save function\
-   Meeting statistics chart\
-   Conversational retrieval chatbot

The theme is customized to an **orange Soft UI**.

------------------------------------------------------------------------

## 🚀 Running the Application

Install dependencies:

``` bash
pip install -r requirements.txt
```

Add your `.env` file:

    OPENAI_API_KEY=xxxxx
    DEEPSEEK_API_KEY=xxxxx

Launch the UI:

``` bash
python main.py
```

------------------------------------------------------------------------

## 🧩 Technologies Used

-   **OpenAI GPT‑4o / GPT‑4o-mini**
-   **DeepSeek Chat Model**
-   **LangChain**
-   **ChromaDB**
-   **Gradio**
-   **Plotly**
-   **t‑SNE (scikit-learn)**

------------------------------------------------------------------------

## ⭐ Features Summary

  Feature          Description
  ---------------- ---------------------------------------------
  Summarization    GPT or DeepSeek with strict business format
  Translation      Multi-language, structure-preserving
  Auto‑Save        Structured markdown files
  Vector Search    MMR retrieval with metadata
  Conversation     Chat about previous meetings
  Visualizations   t-SNE embedding maps
  Dashboard        Monthly meeting count chart

------------------------------------------------------------------------
