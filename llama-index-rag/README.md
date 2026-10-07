# LlamaIndex RAG and customer support agent

Run these commands from the repository root. The project uses Python 3.13 and
`uv` to install the dependencies in `pyproject.toml`:

```bash
uv sync
```

Create a `.env` file in the repository root (or set these environment variables
in your shell):

```dotenv
CEREBRAS_API_KEY=your_cerebras_api_key
OPENAI_API_KEY=your_openai_api_key
TO_EMAIL=sales@example.com

# Use Resend for lead emails:
USE_RESEND=True
RESEND_API_KEY=your_resend_api_key
RESEND_FROM_EMAIL="Agentrixx <info@comms.agentrixx.com>"

# Or use SendGrid instead:
# USE_RESEND=False
# SENDGRID_API_KEY=your_sendgrid_api_key
# SENDGRID_FROM_EMAIL=verified_sender@example.com
```

Keep the `.env` file private; it contains API keys.

`CEREBRAS_API_KEY` is used for chat. LlamaIndex uses its default OpenAI
embedding model when building and searching the index, so `OPENAI_API_KEY` is
also needed for the RAG scripts and the agent's service search. When
`USE_RESEND=True`, lead emails use `RESEND_API_KEY`, `RESEND_FROM_EMAIL`, and
`TO_EMAIL`. Otherwise, they use `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL`, and
`TO_EMAIL`. The sender address must be verified with the selected provider.

## Run the scripts

| Script | Command | What it does |
| --- | --- | --- |
| `Llama_Index_basics.py` | `uv run python Llama_Index_basics.py` | Loads the index from `storage/`, or builds it from `data/` if `storage/` is absent, then runs a sample chat query. |
| `customer_support_agent.py` | `uv run python customer_support_agent.py` | Starts an interactive terminal chat. Type `exit` or `quit` to stop. |
| `openai_agent/customer_support.py` | `uv run python -m openai_agent.customer_support` | Starts the OpenAI Agents SDK version with conversation, search planning, assessment, and email agents. |
| `streamlit_app.py` | `uv run streamlit run streamlit_app.py` | Starts the browser chat app. Open the local URL printed by Streamlit. |
| `main.py` | `uv run python main.py` | Prints a simple hello message to check the Python setup. |

The terminal and Streamlit agents read `data/agentrixx_service_document.md`
for verified services and project examples. `Llama_Index_basics.py` separately
uses the persisted index in `storage/`; if it is missing, run that script to
create it from `data/`.

## OpenAI Agents SDK customer support

Run `uv run python -m openai_agent.customer_support` from the repository root.
Set `OPENAI_API_KEY` for the Agents SDK and the LlamaIndex embedding model. You
can override the default `gpt-5.4-mini` model with `OPENAI_AGENT_MODEL`. The
workflow searches the persisted LlamaIndex vector index in `storage/`; build
that index with `Llama_Index_basics.py` if it is absent. Website scraping uses
the existing public-site scraper in `tools.py`.

The conversation agent chooses whether to reply, assess a need, or hand a
customer's email request to the email agent. The search planner writes vector
queries, the search agent retrieves local index excerpts, and the assessment
agent writes the answer. The email agent has tools for sending a sales summary
and a customer follow-up. The session keeps conversation history in memory
until the process exits.

`tools.py` contains the agent's helper functions. Running
`uv run tools.py` directly sends a sample email to `TO_EMAIL`; normal agent
use does not require running it separately.

When the agent finds a clear service match and the prospect has supplied an
email address, it notifies the sales team at `TO_EMAIL` and sends the prospect
a personalized email with `send_email_to_client(client_email, subject, body)`.
Both emails use the configured sender and email provider.

If Resend returns HTTP 403, the script prints the provider's error message.
Check that `RESEND_FROM_EMAIL` uses a domain verified for sending in the same
Resend account as `RESEND_API_KEY`. Resend's `resend.dev` testing sender can only
send to the account's own email address. To send to `TO_EMAIL` at another
address, verify a domain in Resend and use an address on that domain. You can
also inspect the failed request in Resend's API logs.
