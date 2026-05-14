# AI News Generator

**AI News Generator** is a data processing and analysis project designed to work with Polish news articles. It utilizes Google's Generative AI (Gemini models) via the LangChain library to analyze, categorize, and process scraped news content. 

The main workflow involves reading collected articles, using LLMs to extract key information (such as the main event, topic, and article length), and saving the structured output for further analysis.

## Project Structure

This repository contains the following key files and directories:

### Data Files
*   `news.sqlite3`: An SQLite database storing the collected/scraped news articles.
*   `scraped_news.json`: A JSON export of the scraped news data.
*   `scraped_news_from_db.json`: JSON output created by converting the SQLite database records.
*   `articles_topics.json`: Raw JSON output containing the analysis results from the LLM.
*   `articles_topics_processed.json`: Processed and structured JSON containing formatted insights (ID, Provider, URL, Event, Topic, Length) for each article.

### Scripts and Notebooks
*   `generate_articles.py`: A Python script demonstrating how to connect to the Google Gemini API (e.g., `gemini-1.5-flash`) using LangChain for text generation.
*   `get_topics.ipynb`: A Jupyter Notebook that reads unique articles from the database and prompts a Gemini model to analyze the text, identifying the event described, the general topic, and the article's length. The results are parsed and exported to JSON files.
*   `change_sqlite_to_json.ipynb`: A Jupyter Notebook for converting SQLite database records into JSON format.
*   `generate_gemini.ipynb`: A Jupyter Notebook that reads processed article topics and uses a Gemini model to generate new articles based on those topics, saving the output to model-specific directories.

### Model-Specific Output Directories
The project evaluates and stores processing outputs from different versions of the Gemini models in the following directories:
*   `gemini-2.5-flash/`
*   `gemini-3-flash-preview/`
*   `gemini-3.1-flash-lite-preview/`

## Requirements and Configuration

To run the scripts, you will need a valid Google Gemini API key. Ensure that you have the `langchain-google-genai` package installed and your API key is correctly configured in your local environment.
