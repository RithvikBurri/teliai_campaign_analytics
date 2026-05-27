# TeliAI Campaign Analytics Platform

Production analytics pipeline for TeliAI processing 67K+ conversation threads and 83K+ messages across SMS campaigns. Built with Python, LangGraph, and Supabase, with full coverage from raw CSV ingestion through thread reconstruction, LLM-powered sentiment classification, and an interactive analytics dashboard.

## Thread Reconstruction

Raw CSV data provided by TeliAI was fragmented and unordered. Conversation threads were rebuilt by connecting records using thread ID, sorting messages by timestamp, and reconstructing each conversation into the correct sequential order before feeding it into the sentiment analysis pipeline.

## Sentiment Analysis

LLM-powered sentiment classification was applied through the LangGraph pipeline, running Claude-based classification on each reconstructed conversation thread and labeling each interaction as positive, neutral, or negative to surface campaign performance insights at scale.

## Results

Processed 67K+ conversation threads and 83K+ messages from TeliAI's SMS campaign data. The finalized pipeline delivers:

- **67K+** conversation threads processed end to end
- **83K+** messages analyzed through LLM-powered sentiment classification
- Regional response rate analysis by state and area code
- Engagement timing insights by day and hour
- Campaign performance dashboard with filtering by campaign, region, and area code

---

# Setup and Run

Follow these steps to run the project locally.

## 1. Clone the Repository

```bash
git clone https://github.com/RithvikBurri/teliai_campaign_analytics.git
cd teliai_campaign_analytics
```

## 2. Load Sample Data

Sample data is available in the `sample_data/` folder. Load the CSV files into your Supabase instance:

- `campaigns.csv`
- `messages_anonymized.csv`
- `conversation_history_anonymized.csv`

Update your Supabase credentials in `.env` before running.

## 3. Install Required Packages

Python **3.9** or higher is required.

```bash
pip install -r requirements.txt
```

## 4. Run the Pipeline

Start the FastAPI backend:

```bash
uvicorn TeliAI_Analytics_Pipeline:app --reload
```

Then navigate to `http://127.0.0.1:8000/docs` to access the API and trigger analytics processing.

## How It Works

1. **Data Ingestion** — Raw CSV files are loaded into Supabase for structured storage and cross-campaign querying
2. **Thread Reconstruction** — Conversation threads are rebuilt by joining records on thread ID and sorting by timestamp
3. **Sentiment Classification** — LangGraph pipeline routes each thread to Claude for positive, neutral, or negative classification
4. **Analytics** — Metrics are calculated and served through the FastAPI backend
5. **Dashboard** — Campaign insights are displayed with filtering by campaign, state, and area code
