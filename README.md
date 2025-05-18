# AI-Powered FastAPI Application

A robust API-based application built with FastAPI that provides AI assistant capabilities, content generation, and document Q&A using OpenAI's language models.

## Features

- **AI Chat Assistant**: Interact with an AI assistant through a conversational interface
- **Content Generation**: Generate content based on prompts with customizable length
- **Document Q&A**: Upload PDF documents and ask questions about their content
- **Multi-API Key Support**: Fallback mechanism to handle API rate limits and quota issues
- **Responsive Web Interface**: Modern, user-friendly interface for all features

## Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **AI/ML**: LangChain, OpenAI GPT models
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Document Processing**: PyPDF, FAISS for vector search

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key(s)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-fastapi-app.git
   cd ai-fastapi-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your OpenAI API key(s):
   ```
   OPENAI_API_KEY=your-api-key-1,your-api-key-2
   ```
   You can add multiple API keys separated by commas for the fallback mechanism.

### Running the Application

Start the application with:

```bash
uvicorn main:app --reload --port 8003
```

The application will be available at http://localhost:8003

## API Endpoints

- `POST /api/v1/ask`: Ask a question to the AI assistant
- `POST /api/v1/generate`: Generate content based on a prompt
- `POST /api/v1/document-qa`: Upload a document and ask questions about it

## Error Handling

The application includes robust error handling for:

- API quota limitations with automatic fallback to alternative API keys
- Automatic model fallback to less resource-intensive models when needed
- Clear user-facing error messages with suggestions

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key(s) separated by commas | (Required) |

## Project Structure

```
ai-fastapi-app/
├── app/
│   ├── api/
│   │   └── routes.py         # API route definitions
│   ├── core/
│   │   └── config.py         # Application configuration
│   └── service/
│       └── ai_service.py     # AI service logic
├── static/
│   └── index.html            # Frontend interface
├── main.py                   # Application entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## License

MIT

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/) 