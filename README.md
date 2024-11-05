# Simple Q&A RAG System

This is a simple Question-Answering system using Retrieval Augmented Generation (RAG) with Groq's LLM.

## Setup

1. Make sure you have Python 3.8+ installed.

2. Install the required packages:

```bash
pip install langchain-community langchain-groq langchain sentence-transformers faiss-cpu python-dotenv
```

3. Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

4. Replace `your_groq_api_key_here` with your actual Groq API key.

## Usage

1. Run the script:

```bash
python rag.py
```

2. When prompted, enter the path to your text file.

3. Ask questions about the content of your text file.

4. The system will display:

    - The answer to your question
    - The relevant source documents used to generate the answer

5. Type 'quit' to exit the program.

## How it Works

1. The system loads your text file and splits it into smaller chunks using RecursiveCharacterTextSplitter
2. It creates embeddings for these chunks using sentence-transformers (all-MiniLM-L6-v2 model)
3. The embeddings are stored in a FAISS vector store for efficient retrieval
4. When you ask a question, the system:
    - Finds the most relevant text chunks using FAISS
    - Sends these chunks along with your question to Groq's LLM
    - Returns the LLM's response and source documents

## Features

-   Uses Groq's Mixtral-8x7b model for high-quality responses
-   Efficient vector similarity search with FAISS
-   Local embedding generation with sentence-transformers
-   Custom prompt template for better context handling
-   Display of source documents used for each answer
-   Interactive command-line interface

## Troubleshooting

If you encounter import errors, make sure you have installed all the required packages:

```bash
pip install langchain-community langchain-groq langchain sentence-transformers faiss-cpu python-dotenv
```

For any other issues, check that:

1. Your Groq API key is correctly set in the .env file
2. The text file path you provide exists and is accessible
3. You have a working internet connection for the Groq API calls
