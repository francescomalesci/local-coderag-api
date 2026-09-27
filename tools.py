import logging
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from database import collection

def search_thesis_notes(query: str) -> str:
    """Searches for information in the thesis notes vector database. Use this tool if the question is about the thesis, YOLO, SAM 2, or railway ballast."""
    logging.info(f"Executing TOOL RAG: searching for '{query}'")
    results = collection.query(query_texts=[query], n_results=3)
    if not results['documents'][0]:
        return "No data found in the thesis database."
    return "\n---\n".join(results['documents'][0])

def search_arxiv(query: str) -> str:
    """Searches scientific articles on arXiv. Use this tool ONLY for papers or academic literature."""
    logging.info(f"Executing TOOL arXiv: searching for '{query}'")
    encoded_query = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=2'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        results = [f"- {entry.find('{http://www.w3.org/2005/Atom}title').text.strip()}" for entry in root.findall('{http://www.w3.org/2005/Atom}entry')]
        return "\n".join(results) if results else "No articles found on arXiv."
    except Exception as e:
        return f"Error during arXiv search: {e}"

def save_note(content: str, filename: str = "research_notes.txt") -> str:
    """Saves a summary or note to a local text file. Use this tool when the user explicitly asks to save, annotate, or store information in a file."""
    logging.info(f"Executing TOOL Save Note: writing to '{filename}'")
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content + "\n\n")
        return f"Action completed: Note successfully saved in the file {filename}."
    except Exception as e:
        return f"Error saving file: {e}"

available_tools = {
    'search_thesis_notes': search_thesis_notes,
    'search_arxiv': search_arxiv,
    'save_note': save_note
}