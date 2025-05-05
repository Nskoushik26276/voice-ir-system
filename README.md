# Voice-Based Information Retrieval System

This Python project accepts an audio file as input, transcribes the speech using Google's Speech Recognition API, searches Wikipedia and the web for relevant content, and summarizes it using Hugging Face Transformers. The output is rendered as an HTML file with clickable links.

## Features

- Speech-to-text conversion (Google Speech Recognition)
- Wikipedia summarization
- Web scraping fallback if no Wikipedia page found
- HuggingFace Transformers summarizer (optional)
- Generates downloadable HTML output

## Installation

```bash
pip install -r requirements.txt
