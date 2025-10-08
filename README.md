# Computer Use Agent Demo

This project automates a Playwright browser while using Gemini's Computer Use API to reason about actions and extract information from the web page.

## Setup

1. Create a virtual environment (optional but recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   # or
   uv pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and provide your Gemini API key:
   ```bash
   cp .env.example .env
   echo GEMINI_API_KEY=your_actual_key >> .env
   ```
3. Playwright also needs browser binaries. Install them once:
   ```bash
   playwright install chromium
   ```

## Running

Execute the agent script:
```bash
uv run computer_use_agent.py
```

By default the script launches Chromium in headed mode to show the agent acting on the page. Set `headless=True` in the script if you prefer headless automation.

## Repository Hygiene

- Sensitive environment values stay out of version control thanks to `.gitignore` and `.env.example`.
- Dependencies are tracked both in `requirements.txt` for quick installs and `computer/pyproject.toml` for packaging the `computer` module.
