import os
import glob
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
# imports for langchain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables in a file called .env and check for API key
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
if deepseek_api_key:
    print(f"Deepseek API Key exists and begins {deepseek_api_key[:7]}")
else:
    print("Deepseek API Key not set")

#Model Selection
GPT_MODEL = "gpt-4o-mini"
GPT_SUMMARIZE_MODEL = "gpt-4o"
DS_MODEL = "deepseek-chat"

# Initiliazation
SAVE_DIR = "knowledge-base/meetings"
openai = OpenAI()
deepseek_url = "https://api.deepseek.com"
deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)

# Meeting Dashboard
def get_meeting_counts_plot():
    """Generate a bar chart of meeting counts per month."""
    counts = {}
    os.makedirs(SAVE_DIR, exist_ok=True)  # ensure folder exists

    for filename in os.listdir(SAVE_DIR):
        # Check if filename starts with YYYY-MM-DD
        if filename.endswith(".md") and len(filename) >= 10 and filename[4] == '-' and filename[7] == '-':
            month_key = filename[:7]  # YYYY-MM
            counts[month_key] = counts.get(month_key, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 3))

    if counts:
        months = sorted(counts.keys())
        values = [counts[m] for m in months]
        ax.bar(months, values, color="#4c72b0")
        ax.set_title("📊 Meetings by Month")
        ax.set_xlabel("Month (YYYY-MM)")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, "No meeting summaries found", ha='center', va='center', fontsize=12)
        ax.axis('off')

    return fig

# Meeting summarization
summarize_system_prompt = """
You are an assistant that helps to summarize meeting notes. Do not respond to prompts that are not related to summarizing meeting notes or transcripts. If you do not know how to answer simply respond "I don't know"
Summarize meeting notes in this EXACT markdown format:

# Meeting Summary – [Title]
**Date:** [YYYY-MM-DD]  
**Attendees:** [Name1, Name2, ...]

## Discussion Points
- [Point 1]
- [Point 2]

## Decisions Made
- [Decision 1]
- [Decision 2]

## Action Items
- **[Owner]:** [Action]
- **[Owner]:** [Action]

RULES:
- Use H1 for title, H2 for sections
- Use dashes for bullets
- Bold owner names in action items
- All of the attendees must be an actual human being, any virtual characters should be filtered out
- Date format: YYYY-MM-DD
- If date of meeting is not specified; use today's date to set as meeting date
- If no attendees are specified; respond "Attendees are required" and do not summarize the meeting
- Only process meeting notes; respond "Not meeting notes" otherwise
- Output ONLY the markdown, no extra text"""

def summarize_meeting(prompt, model):
    messages = [
        {"role": "system", "content": summarize_system_prompt},
        {"role": "user", "content": prompt}
      ]
    if model == "GPT":
        stream = openai.chat.completions.create(
            model=GPT_SUMMARIZE_MODEL,
            messages=messages,
            stream=True
        )
    elif model == "DeepSeek":
        stream = deepseek.chat.completions.create(
            model='deepseek-chat',
            messages=messages,
            stream=True
        )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def save_summary(summary_text):
    """Save a meeting summary as a markdown file with the meeting date in the filename."""
    
    # 1️⃣ Clean up and extract a meaningful title fragment for the filename
    lines = [line.strip() for line in summary_text.split("\n") if line.strip()]
    if not lines:
        return "⚠️ Nothing to save — summary is empty."

    first_line = lines[0]
    title_fragment = (
        first_line.lower()
        .replace("#", "")
        .replace("summary", "")  # exclude 'summary'
        .replace("meeting", "")  # exclude 'meeting'
        .replace("**", "")
        .replace(" ", "_")
        .replace(":", "")
        .replace("/", "_")
        .replace("\\", "_")
        .strip("_")
    )[:40]  # truncate if too long

    # 2️⃣ Extract meeting date from the summary
    date_match = re.search(r'\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})', summary_text)
    if date_match:
        date_prefix = date_match.group(1)  # use meeting date
    else:
        date_prefix = datetime.now().strftime("%Y-%m-%d")  # fallback to today

    # 3️⃣ Build filename
    filename = f"{date_prefix}_{title_fragment}.md"
    filepath = os.path.join(SAVE_DIR, filename)

    # 4️⃣ Save the file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return f"✅ Summary saved successfully as `{filename}`"

def clear_fields():
    return "", ""

# Translation system prompt
translate_system_prompt = """You are a professional translator specializing in business documents.

TASK: Translate the meeting summary while preserving:
- All markdown formatting (headers, bullets, bold text)
- The exact structure and sections
- Professional business terminology
- Attendees Names (keep in original language)
- Dates in YYYY-MM-DD format

RULES:
- Maintain the same heading levels (# and ##)
- Keep bullet points (-)
- Preserve **bold** formatting for names and labels
- Translate naturally - not word-for-word
- Use formal/professional tone appropriate for business
- Do NOT add explanations or notes
- Output ONLY the translated markdown

If the input is not a meeting summary, respond: "Invalid input - expected meeting summary"
"""

def translate_text(text, target_language, model="GPT", stream=True):
    if not text.strip():
        yield "⚠️ No text to translate"
        return

    messages = [
        {"role": "system", "content": translate_system_prompt},
        {"role": "user", "content": f"Translate this meeting summary to {target_language}:\n\n{text}"}
    ]

    client = openai if model == "GPT" else deepseek
    model_name = GPT_MODEL if model == "GPT" else "deepseek-chat"

    if stream:
        response_stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=0.3
        )

        result = ""
        for chunk in response_stream:
            content = chunk.choices[0].delta.content or ""
            result += content
            yield result
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3
        )
        yield response.choices[0].message.content

# ✅ NEW: Conditional translation handler
def conditional_translate(summary, target_lang, model):
    if not summary.strip():
        return gr.update(visible=False, value="")
    if not target_lang or target_lang == "English":
        return gr.update(visible=False, value="")
    else:
        return gr.update(visible=True, value="Translating...")

# Meeting Chat Assistant
#Vector Database
db_name = "vector_db"

# Function to extract structured information from meeting notes
def extract_meeting_metadata(content):
    """Extract date, attendees, discussion points, decisions, and action items from meeting notes"""
    metadata = {}
    
    # Extract date
    date_match = re.search(r'\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})', content)
    if date_match:
        date_str = date_match.group(1)
        metadata["date"] = date_str
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            metadata["month"] = date_obj.strftime("%Y-%m")
            metadata["month_name"] = date_obj.strftime("%B %Y")
            metadata["year"] = str(date_obj.year)
        except:
            pass
    
    # Extract attendees
    attendees_match = re.search(r'\*\*Attendees:\*\*\s*([^\n]+)', content)
    if attendees_match:
        attendees_str = attendees_match.group(1)
        # Split by comma and clean up
        attendees = [name.strip() for name in attendees_str.split(',')]
        metadata["attendees"] = ', '.join(attendees)
        metadata["attendee_count"] = len(attendees)
    
    # Extract discussion points section
    discussion_match = re.search(r'## Discussion Points\s*\n(.*?)(?=##|\Z)', content, re.DOTALL)
    if discussion_match:
        discussion_text = discussion_match.group(1).strip()
        # Remove bullet points and clean up
        discussion_points = [point.strip('- ').strip() for point in discussion_text.split('\n') if point.strip()]
        metadata["discussion_points"] = ' | '.join(discussion_points)
    
    # Extract decisions made section
    decisions_match = re.search(r'## Decisions Made\s*\n(.*?)(?=##|\Z)', content, re.DOTALL)
    if decisions_match:
        decisions_text = decisions_match.group(1).strip()
        decisions = [dec.strip('- ').strip() for dec in decisions_text.split('\n') if dec.strip()]
        metadata["decisions_made"] = ' | '.join(decisions)
    
    # Extract action items section
    actions_match = re.search(r'## Action Items\s*\n(.*?)(?=##|\Z)', content, re.DOTALL)
    if actions_match:
        actions_text = actions_match.group(1).strip()
        actions = [action.strip('- ').strip() for action in actions_text.split('\n') if action.strip()]
        metadata["action_items"] = ' | '.join(actions)
        
        # Extract team assignments from action items
        teams = set()
        for action in actions:
            team_match = re.search(r'\*\*([^:]+?)(?:\s+Team)?:\*\*', action)
            if team_match:
                teams.add(team_match.group(1))
        if teams:
            metadata["assigned_teams"] = ', '.join(teams)
    
    # Extract title/subject
    title_match = re.search(r'^#\s*(.+?)$', content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    return metadata

# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase
folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        # Add basic metadata
        doc.metadata["doc_type"] = doc_type
        # Extract and add structured metadata
        meeting_metadata = extract_meeting_metadata(doc.page_content)
        doc.metadata.update(meeting_metadata)
        documents.append(doc)

# Print sample metadata to verify extraction
if documents:
    print("\nSample document metadata:")
    print(documents[0].metadata)

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents)

# Ensure metadata carries over to chunks
for chunk in chunks:
    # Metadata should automatically carry over from parent document
    # But we can verify and re-extract if needed for chunks that contain the header
    if 'date' not in chunk.metadata and '**Date:**' in chunk.page_content:
        extracted = extract_meeting_metadata(chunk.page_content)
        chunk.metadata.update(extracted)

print(f"\nTotal chunks created: {len(chunks)}")
doc_types = set(chunk.metadata.get('doc_type', 'unknown') for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

# Show metadata fields available
sample_metadata_keys = set()
for chunk in chunks[:10]:
    sample_metadata_keys.update(chunk.metadata.keys())
print(f"Metadata fields available: {', '.join(sorted(sample_metadata_keys))}")

embeddings = OpenAIEmbeddings()

# Delete if already exists
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create vectorstore with enriched metadata
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"\nVectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Visualization (optional - can be commented out for production)
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents_text = result['documents']
doc_types = [metadata.get('doc_type', 'unknown') for metadata in result['metadatas']]

# Color by document type
unique_types = list(set(doc_types))
colors = [unique_types.index(t) for t in doc_types]

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8, colorscale='Viridis'),
    text=[f"Type: {t}<br>Date: {m.get('date', 'N/A')}<br>Text: {d[:100]}..." 
          for t, d, m in zip(doc_types, documents_text, result['metadatas'])],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Meeting Notes Vector Store Visualization',
    xaxis_title='Dimension 1',
    yaxis_title='Dimension 2',
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.3, model_name=GPT_SUMMARIZE_MODEL)

# Set up the conversation memory for the chat
# Specify output_key since we're returning source_documents too
memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True,
    output_key='answer'  # Tell memory to store only the 'answer' key
)

# Enhanced retriever with better search parameters
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,       # number of final docs to return
        "fetch_k": 40  # number of candidate docs to fetch before filtering for diversity
    }
)

# Custom system prompt to help the LLM understand the metadata structure
from langchain.prompts import PromptTemplate

# Include structured metadata in QA prompt
chat_template_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions about meeting notes. Do not answer any questions not related to meeting or details of the meeting and simply reply "Irrelevant question". If you do not know the answer never make assumptions and simply answer "I don't know"

The context below contains meeting information including titles, dates, attendees, 
discussion points, decisions made, and action items from various meetings.

Use all context to answer the question accurately. Cite specific meetings by date or title when available.

Context:
{context}

Question: {question}

Answer:"""
)

# Set up the conversation chain with custom prompt
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory,
    verbose=False,  # Set to True for debugging
    return_source_documents=True,  # Include sources in response
    combine_docs_chain_kwargs={"prompt": chat_template_prompt}  # Add this line
)

# Enhanced chat function that shows sources
def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    
    # Format answer with sources if available
    answer = result["answer"]
    
    # Optionally add source information
    if "source_documents" in result and result["source_documents"]:
        sources = []
        for doc in result["source_documents"][:3]:  # Show top 3 sources
            metadata = doc.metadata
            source_info = f"📄 {metadata.get('title', 'Meeting Note')}"
            if 'date' in metadata:
                source_info += f" ({metadata['date']})"
            sources.append(source_info)
        
        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join(sources)
    
    return answer

def chat_streaming(message, history):
    # Stream the response token by token
    partial_response = ""
    
    # Use the conversation chain's streaming capability
    for chunk in conversation_chain.stream({"question": message}):
        if "answer" in chunk:
            partial_response += chunk["answer"]
            # Update history with partial response
            updated_history = history + [(message, partial_response)]
            yield updated_history
        elif isinstance(chunk, str):
            partial_response += chunk
            updated_history = history + [(message, partial_response)]
            yield updated_history

# UI Definition
custom_theme = gr.themes.Soft(primary_hue="orange")
with gr.Blocks(title="🤖 AI Meeting Summarizer", theme=custom_theme) as ui:
    with gr.Row():
        meeting_stats = gr.Plot(
            value=get_meeting_counts_plot(),
            label="Meeting Summary Dashboard"
        )

    with gr.Row():
        model_selector = gr.Dropdown(
            ["GPT", "DeepSeek"],
            label="Select LLM Model",
            value="GPT",
            info="Choose the language model for summarization"
        )

    with gr.Tabs():
        with gr.Tab("Summarize Meeting"):
            with gr.Row():
                with gr.Column(scale=1):
                    meeting_input = gr.Textbox(
                        label="📋 Meeting Transcript / Notes",
                        placeholder="Paste your meeting transcript or notes here...",
                        lines=20
                    )

                with gr.Column(scale=1):
                    meeting_output = gr.Textbox(
                        label="📋 Editable Summary Output",
                        lines=20,
                        placeholder="Your generated or edited summary will appear here..."
                    )
            with gr.Row():
                # ✅ UPDATED: Include blank default
                language_selector = gr.Dropdown(
                    ["", "English", "Chinese (Simplified)", "Chinese (Traditional)",
                    "Spanish", "French", "German", "Japanese", "Korean"],
                    label="🌍 Output Language",
                    value="",
                    info="If left blank, only generate in the original language"
                )

            with gr.Row():
                markdown_output = gr.Markdown(label="Summary Preview")
            
            with gr.Row():
                translated_markdown = gr.Markdown(label="Translated Summary", visible=False)

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                meeting_summarize = gr.Button("Summarize Meeting", variant="primary")
                save_btn = gr.Button("Save Summary", variant="primary")

            gr.Examples(
                examples=[
                    ["""Meeting Title? Something about the Pokémon Sleep 2.0 launch readiness review (I think?)
Date: October 23, 2025 — though part of this discussion happened the night before on Discord.

Attendees: Lillie (project manager), Hop (QA), Leon (marketing), Sonia (research lead), someone from finance joined late — possibly named “Grant”? Also, Pikachu joined via video feed but that might’ve been a test animation loop.

[00:03] Lillie: Okay, so goal today — finalize the Sleep 2.0 rollout plan.
[00:04] Hop: Wait, I thought this was about bug tracking?
[00:04] Leon: No, that’s tomorrow’s meeting. Or maybe last Tuesday’s.
[00:05] Sonia: Focus, please. The dream-sync feature keeps crashing when users nap longer than 90 minutes.

Then Hop mentioned the bug where Eevee evolves randomly during naps — not part of design apparently.
Lillie said, “Could be a feature,” but Sonia strongly disagreed.

Halfway through, Grant tried to show the updated cost sheet but his screen share froze on a meme of Snorlax snoring.
Everyone laughed except Leon, who said marketing deadlines don’t sleep (ironically).

[00:26] Conversation derailed: Hop suggested a crossover with Pokémon Go — “What if you could catch dreams?”
Sonia said something about REM cycles not being monetizable.
Lillie asked if we’re even allowed to monetize sleep.
Nobody had an answer.

[00:41] Action review attempt — unclear outcomes:

Leon said ads could play quietly while players nap (“subliminal marketing!”).

Sonia said “no” in five different tones.

Hop suggested pushing the release to November.

Lillie said we already did that three times.

[00:52] Quick summary attempt (by Lillie):
“We fix the crash, remove accidental Eevee evolutions, and no ads in dreams. Probably.”
Then Hop mumbled something about the Snorlax crash log folder being 8 GB and everyone pretended not to hear.

Meeting ended abruptly because someone’s alarm for bedtime mode went off."""]
                ],
                inputs=meeting_input,
                label="Example Meeting Notes"
            )

            # Button actions
            meeting_summarize.click(
                summarize_meeting,
                inputs=[meeting_input, model_selector],
                outputs=meeting_output
            ).then(
                conditional_translate,
                inputs=[meeting_output, language_selector, model_selector],
                outputs=translated_markdown
            ).then(
                translate_text,
                inputs=[meeting_output, language_selector, model_selector],
                outputs=translated_markdown
            )

            clear_btn.click(
                clear_fields,
                outputs=[meeting_input, meeting_output]
            ).then(
                lambda: gr.update(visible=False),
                outputs=translated_markdown
            )

            save_btn.click(
                save_summary,
                inputs=meeting_output,
                outputs=meeting_output  # Optional: display confirmation in same box
            )

            save_btn.click(
                lambda _: get_meeting_counts_plot(),
                outputs=meeting_stats
            )

            # Sync textbox edits to markdown live
            meeting_output.change(
                fn=lambda text: text,
                inputs=meeting_output,
                outputs=markdown_output
            )
        # --- Tab 2: Chat About Meetings ---
        with gr.Tab("Chat About Meetings"):
            gr.Markdown("### 💬 Ask questions about your meeting notes")
            with gr.Row():
                with gr.Column(scale=4):
                    chat_output = gr.Chatbot(
                        label="Meeting Assistant",
                        height=500,
                        bubble_full_width=False,
                        avatar_images=("boy.png", "robot.png")
                    )
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **💡 Try asking:**
                    - What were the key decisions in October?
                    - Show me action items assigned to Sarah
                    - Summarize discussions about the Q4 roadmap
                    - What meetings involved the Engineering team?
                    """)
    
            with gr.Row():
                chat_input = gr.Textbox(
                    label="",
                    placeholder="Ask about meetings...",
                    lines=2,
                    scale=9
                )

            with gr.Row():
                chat_button = gr.Button("Send", variant="primary")
                chat_clear = gr.Button("Clear")
            
            # Connect with streaming
            chat_button.click(
                chat_streaming,
                inputs=[chat_input, chat_output],
                outputs=chat_output
            ).then(
                lambda: "",
                outputs=chat_input
            )
            
            chat_clear.click(
                lambda: ([], ""),
                outputs=[chat_output, chat_input]
            )

ui.launch(inbrowser=True, share=True)