!pip install SpeechRecognition transformers numpy pydub wikipedia-api googlesearch-python requests beautifulsoup4
!apt-get install ffmpeg

import speech_recognition as sr
from transformers import pipeline
from collections import deque
from pydub import AudioSegment
import wikipediaapi
import re
from googlesearch import search
import requests
from bs4 import BeautifulSoup

recognizer = sr.Recognizer()
qa_pipeline = pipeline('question-answering')
context_history = deque(maxlen=5)

def listen_for_query_from_audio(audio_file):
    try:
        if not audio_file.endswith('.wav'):
            audio_file = convert_audio_to_wav(audio_file)

        with sr.AudioFile(audio_file) as source:
            print("Processing audio file for query...")
            audio = recognizer.record(source)
            print("Recognizing...")
            query = recognizer.recognize_google(audio)
            return query
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Error with Google Speech Recognition: {e}")
        return None

def sanitize_query(query):
    return re.sub(r"[^a-zA-Z0-9\s]", "", query).strip()

def retrieve_wikipedia_info(query):
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent="MyPythonApp/1.0 (contact: myemail@example.com)",
            language="en"
        )

        page = wiki.page(query)
        if page.exists():
            return page.summary[:500], page.fullurl

        print(f"Page '{query}' does not exist. Searching alternative Wikipedia page...")
        wiki_search_results = google_search(f"{query} site:en.wikipedia.org")

        for link in wiki_search_results:
            if "wikipedia.org" in link:
                title = link.split("/")[-1].replace("_", " ")
                page = wiki.page(title)
                if page.exists():
                    return page.summary[:500], link

        return None, None
    except Exception as e:
        print(f"Error retrieving from Wikipedia: {e}")
        return None, None

def google_search(query):
    try:
        results = list(search(query, num_results=3))
        return [url for url in results if url.startswith("http")]
    except Exception as e:
        print(f"Google search error: {e}")
        return []

def fetch_additional_info(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs[:3]])
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return "No additional content found."

def convert_audio_to_wav(audio_file):
    print("Converting audio to WAV format...")
    audio = AudioSegment.from_file(audio_file)
    wav_file = audio_file.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_file, format="wav")
    return wav_file

def generate_html_output(query, relevant_info, page_url, alternative_links, additional_info=None):
    # Generate HTML content for output
    html_content = f"<html><head><title>Query Result</title></head><body>"
    html_content += f"<h1>Query: {query}</h1>"

    if relevant_info:
        html_content += f"<h2>Information from Wikipedia:</h2><p>{relevant_info}</p>"
        html_content += f"<p>For more details, visit: <a href='{page_url}' target='_blank'>{page_url}</a></p>"

    if alternative_links:
        html_content += f"<h2>Alternative Sources:</h2><ul>"
        for link in alternative_links:
            html_content += f"<li><a href='{link}' target='_blank'>{link}</a></li>"
        html_content += "</ul>"

    if additional_info:
        html_content += f"<h2>Additional Information:</h2><p>{additional_info}</p>"

    html_content += "</body></html>"

    # Write the HTML content to a file
    html_filename = "query_output.html"
    with open(html_filename, "w") as file:
        file.write(html_content)

    return html_filename

def ir_system(audio_file):
    query = listen_for_query_from_audio(audio_file)
    if not query:
        print("No query recognized.")
        return

    print(f"Query: {query}")
    relevant_info, page_url = retrieve_wikipedia_info(query)

    # Initialize alternative_links to ensure it's always defined
    alternative_links = []

    if relevant_info:
        print(f"Information: {relevant_info}")
    else:
        print("No Wikipedia page found. Searching alternative sources...")
        alternative_links = google_search(query)

        if alternative_links:
            print("Alternative links:")
            for idx, link in enumerate(alternative_links, 1):
                print(f"{idx}. {link}")
                additional_info = fetch_additional_info(link)
                print(f"Extra info from {link}: {additional_info}")
        else:
            print("No alternative sources found.")

    # Generate HTML output and download it
    html_filename = generate_html_output(query, relevant_info, page_url, alternative_links)
    from google.colab import files
    files.download(html_filename)

# Upload the audio file for processing
from google.colab import files

uploaded = files.upload()

audio_file = next(iter(uploaded))
ir_system(audio_file)
