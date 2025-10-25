# stock-agent

Quick notes to set up the project's Python virtual environment and install dependencies (macOS / zsh).

## Create & activate a venv

From the project root:

```bash
# create a virtual environment named `venv`
python3 -m venv venv

# activate it (zsh)
source venv/bin/activate
```

When the venv is active your prompt will usually include `(venv)`. If you prefer not to activate the venv, you can run the project Python directly with `/path/to/project/venv/bin/python`.

## Upgrade pip and install dependencies

With the venv activated (recommended):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or without activating the venv (explicit path):

```bash
/Users/youruser/Documents/CodingProjects/stock-agent/venv/bin/python -m pip install --upgrade pip
/Users/youruser/Documents/CodingProjects/stock-agent/venv/bin/python -m pip install -r /Users/youruser/Documents/CodingProjects/stock-agent/requirements.txt
```

Replace `/Users/youruser/Documents/CodingProjects/stock-agent` with your local path if different.

## Regenerate `requirements.txt`

If you add or update packages, regenerate the pinned list from the active venv:

```bash
# recommended while venv is active
python -m pip freeze > requirements.txt

# or explicitly
/path/to/venv/bin/python -m pip freeze > requirements.txt
```

## Running the project

Run the project entrypoint (`main.py`) or the fetch script directly. Both support `--json` and `--tickers` flags.

```bash
# run from project root (with venv activated)
python main.py --json

# or run the fetch script directly
python src/fetch_prices.py --json

# pass a custom ticker list
python main.py --tickers AAPL,MSFT --json
```

If you prefer not to activate the venv, substitute the full venv python path.

## Environment variables

`src/fetch_news.py` uses `python-dotenv` and expects a `NEWSAPI_KEY` variable if you use the news fetcher. Create a `.env` file in the project root with:

```env
NEWSAPI_KEY=your_api_key_here
```

## Troubleshooting

- "pip: command not found": use `python -m pip` with the desired Python executable or activate the venv.
- If network or API errors occur when fetching prices, check your network and rate limits. Consider adding retries/backoff if needed.

If you want, I can add a small `Makefile` or shell helper scripts to make the common operations (create venv, install, run) one-command shortcuts.
