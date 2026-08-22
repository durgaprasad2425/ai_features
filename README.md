# Kodefast AI Features

A FastAPI service for AI-powered text analysis, image-to-content generation, AI image generation, and MongoDB-backed chat sessions.

The project has been reorganized into an MVP structure. Existing clients can continue using the original `aifields:app3` entrypoint and API paths.

## Features

- Text summarization with Gemini
- Marketing and general content generation
- Keyword extraction
- Sentiment analysis
- Image analysis from one or more public image URLs
- AI image generation with OpenAI
- Optional logo download, resizing, positioning, and overlay
- MongoDB-backed conversation history
- Static serving for generated PNG images
- Automatic OpenAPI documentation through FastAPI

## Requirements

- Python 3.10 or newer
- MongoDB connection string for text-session history
- Google Gemini API key for text and image-content features
- OpenAI API key for image generation
- Publicly reachable image URLs for image-to-content requests

## Project Layout

The repository root contains the environment and the new MVP package. The existing application directory contains the compatibility entrypoint.

```text
E:\ai_features-main\
|-- .env                         Local secrets; ignored by Git
|-- .env.example                 Safe environment template
|-- .gitignore
|-- .venv\                       Local virtual environment; ignored by Git
|-- requirements.txt
|-- app\                         Refactored MVP package
|   |-- main.py                  Creates app3 and mounts /static
|   |-- api\routes.py            POST / and POST /generate_content_via_urls/
|   |-- core\
|   |   |-- config.py            Loads environment configuration
|   |   |-- providers.py          Creates optional Gemini/OpenAI clients
|   |   `-- storage.py            Defines relative static image storage
|   |-- models\requests.py       Pydantic request models
|   |-- services\
|   |   |-- chat.py               MongoDB history and session management
|   |   |-- image_content.py      Gemini image-to-content workflow
|   |   `-- image_generation.py   OpenAI image generation workflow
|   `-- utils\
|       |-- images.py             Download, filename, logo, and overlay helpers
|       `-- prompts.py            LangChain prompt templates and mappings
|-- docs\mvp-structure.md        Architecture and migration notes
|-- tests\                       Test location
`-- ai_features-main\
    |-- aifields.py              Compatibility entrypoint: aifields:app3
    |-- README.md
    `-- static\images\          Generated image output when run here
```

## Installation on Windows

Open PowerShell in the repository root:

```powershell
Set-Location E:\ai_features-main
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the virtual environment is already created, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The execution-policy change applies only to the current PowerShell process. It does not change the machine-wide policy.

## Environment Configuration

Copy `.env.example` to `.env` in the repository root and replace the placeholder values. Keep `.env` private.

```powershell
Copy-Item .env.example .env
```

Required variables:

```env
AI_CONNECTION_STRING=mongodb+srv://user:password@cluster.example.com
DATABASE_NAMEE=kodefast_db
BASE_URL=http://localhost:8000/static/images
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Variable usage:

| Variable | Used by | Purpose |
| --- | --- | --- |
| `AI_CONNECTION_STRING` | Chat service | MongoDB connection for session history |
| `DATABASE_NAMEE` | Chat service | MongoDB database name |
| `BASE_URL` | Image route | Public base URL returned for saved images |
| `GOOGLE_API_KEY` | Gemini provider | Text generation and image understanding |
| `OPENAI_API_KEY` | OpenAI provider | DALL-E image generation |

The application can start and expose `/docs` without provider keys. Requests that require an unconfigured provider return a configuration error instead of failing during module import.

## Running the API

From the repository root, activate `.venv` and run:

```powershell
python -m uvicorn --app-dir .\ai_features-main aifields:app3 --reload --host 0.0.0.0 --port 8000
```

The compatibility entrypoint imports the refactored modules and exposes the same `app3` object. The `--app-dir` option tells Uvicorn where `aifields.py` is located.

An equivalent command from the application directory is:

```powershell
Set-Location .\ai_features-main
python -m uvicorn aifields:app3 --reload --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at:

- http://localhost:8000/docs
- http://localhost:8000/redoc
- http://localhost:8000/openapi.json

If port 8000 is already in use, choose another port, for example `--port 8001`.

If Uvicorn reports `Could not import module "aifields"`, run the root-level command with `--app-dir .\ai_features-main` or change into the nested application directory first.

## API Endpoints

### `POST /`

Handles text features and AI image generation.

Request body:

```json
{
  "Enter_your_prompt": "A short product review to analyze",
  "session_id": "session-123",
  "prompt_type": "sentiment_analysis",
  "flag": false
}
```

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `Enter_your_prompt` | string | Prompt or content to process; cannot be blank |
| `session_id` | string | MongoDB chat-history session identifier |
| `prompt_type` | string | Selects the AI workflow |
| `flag` | boolean | Clears the current session history when true |

Supported text prompt types:

- `summary`
- `content_generation`
- `keyword_extraction`
- `sentiment_analysis`
- `default`

Text response example:

```json
{
  "data": "Positive"
}
```

Keyword response example:

```json
{
  "data": ["build quality", "delivery", "review"]
}
```

For `prompt_type: "image_generation"`, the service calls OpenAI, saves the PNG under `static/images/`, and returns an array containing the public image URL:

```json
[
  "http://localhost:8000/static/images/futuristic_city.png"
]
```

The image prompt can include a logo URL. The service removes the URL from the generation prompt, downloads the logo, resizes it, and overlays it on the generated image. Position phrases such as `top-left`, `top-right`, `center`, or `bottom-right` are recognized.

Example PowerShell request:

```powershell
$body = @{
  Enter_your_prompt = "A watercolor city skyline at sunset"
  session_id = "demo-session"
  prompt_type = "content_generation"
  flag = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/ -Method Post -ContentType "application/json" -Body $body
```

### `POST /generate_content_via_urls/`

Analyzes one or more public image URLs with Gemini.

Request body:

```json
{
  "url": [
    "https://example.com/image.jpg"
  ],
  "prompt_type": "content_generation"
}
```

Supported image prompt types:

- `content_generation`
- `sentiment_analysis`
- `keyword_extraction`
- `summarization`

Note that the image endpoint uses `summarization`, while the text endpoint uses `summary`.

Response shape:

```json
{
  "generated_contents": [
    "A cheerful birthday celebration with friends and family."
  ]
}
```

## Compatibility and Migration

The refactor preserves these public contracts:

- Uvicorn target: `aifields:app3`
- `POST /`
- `POST /generate_content_via_urls/`
- `/static` file mounting
- Existing prompt-type names
- Existing response shapes
- Existing image helper imports from `aifields`
- Relative `static/images` runtime behavior

The compatibility file [aifields.py](aifields.py) re-exports the FastAPI app, request models, prompt mappings, services, and image utilities. New development should place functionality in the relevant `app/` module rather than adding more code to the compatibility file.

The recommended migration order is documented in [`../docs/mvp-structure.md`](../docs/mvp-structure.md). Add tests before changing the compatibility entrypoint or public route behavior.

## Validation

From the repository root:

```powershell
python -m compileall -q .\app .\ai_features-main\aifields.py
```

The API can be smoke-tested without AI credentials by opening `/docs`. AI requests require valid provider credentials and, for text sessions, a reachable MongoDB instance.

## Security Notes

- Never commit `.env`, API keys, database passwords, or generated private assets.
- Use restricted API keys with only the permissions required by this service.
- Use HTTPS for deployed `BASE_URL` values and public API access.
- Validate and restrict remote image URLs before exposing this service publicly.
- Add authentication and rate limiting before production deployment.
