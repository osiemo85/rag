"""Tools used by the customer-support agent."""

import html
import os
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.cerebras import Cerebras
import resend
from resend.exceptions import ResendError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


# Match the configuration used when the persisted index was created and queried
# in Llama_Index_basics.py.  In particular, do not override the index's stored
# 1,536-dimension embedding setup with text-embedding-3-small.
load_dotenv(override=True)
Settings.llm = Cerebras(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
)


PERSIST_DIR = "./storage"


@lru_cache(maxsize=1)
def _service_index():
    """Load the service index once per process."""
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)


def get_our_services(prompt: str) -> str:
    """Search Agentrixx's services and projects knowledge base for relevant information.

    Args:
        prompt (str): Short precise prompt of about 7 words.

    Returns:
        str: Relevant service information or a readable search error.
    """
    print(f"------------get_our_services called with prompt: {prompt}--------------")
    if not prompt.strip():
        return "Please provide a question about our services."

    try:
        query_engine = _service_index().as_query_engine()
        # print(f'----query output: {query_engine.query(prompt)} ----')
        return str(query_engine.query(prompt))
    except Exception as error:
        return f"Unable to search the services knowledge base: {error}"


async def search_services_and_projects(prompt: str) -> str:
    """Retrieve Agentrixx services and projects from the local source document.

    Args:
        prompt (str): The question or business need to look up.

    Returns:
        str: The source document containing actual services and projects, or an error.
    """
    if not prompt.strip():
        return "Please provide a question about Agentrixx services or projects."

    document_path = Path(__file__).resolve().parent / "data" / "agentrixx_service_document.md"
    try:
        return document_path.read_text(encoding="utf-8")
    except OSError as error:
        return f"Unable to read the services document: {error}"
class _TextExtractor(HTMLParser):
    """Extract visible text from simple HTML pages."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._ignored_tag:
            self._ignored_tag = None

    def handle_data(self, data: str) -> None:
        if not self._ignored_tag:
            self.parts.append(data)


def scrape_prospect_website(url: str) -> str:
    """Fetch readable text from a prospect's public website.

    Args:
        url (str): A full public URL beginning with ``http://`` or ``https://``.

    Returns:
        str: Up to 6,000 characters of website text or a readable fetch error.
    """
    print(f"--------------scrape_prospect_website called once with url: {url}.................")
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return "Error: provide a full public URL beginning with http:// or https://."

    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LeadQualificationBot/1.0)"})
    try:
        with urlopen(request, timeout=15) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                return f"Error: expected an HTML page, but received {content_type}."
            page = response.read(1_000_000).decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    except HTTPError as error:
        return f"Error: the website returned HTTP {error.code}."
    except URLError as error:
        return f"Error: could not reach the website ({error.reason})."
    except TimeoutError:
        return "Error: the website took too long to respond."

    extractor = _TextExtractor()
    extractor.feed(page)
    text = html.unescape(" ".join(extractor.parts))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Error: no readable text was found on the website."
    return text[:6000]


def notify_sales_team(subject: str, body: str) -> str:
    """Send Agentrixx sales team an email notifying them of a qualified lead.

    Args:
        subject (str): A concise subject describing the qualified lead.
        body (str): Lead contact details, business information, needs, and
            matched Agentrixx services.

    Returns:
        str: A delivery confirmation or a readable configuration or delivery error.
    """
    print(f"--------------notify_sales_team called with subject: {subject} and body: {body}.................")
    to_email = os.getenv("TO_EMAIL")
    if not to_email:
        return "Error: TO_EMAIL must be configured."
    return _send_email(to_email, subject, body, "the sales team has been notified")


def send_email_to_client(client_email: str, subject: str, body: str) -> str:
    """Email a prospective client after their needs match an Agentrixx service.

    Args:
        client_email (str): The email address supplied by the prospective client.
        subject (str): A concise subject about the relevant service.
        body (str): A personalized HTML message summarizing the match and next steps.

    Returns:
        str: A delivery confirmation or a readable configuration or delivery error.
    """
    print(f"--------------send_email_to_client called with client_email: {client_email}, subject: {subject} and body: {body}.................")
    client_email = client_email.strip()
    if not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", client_email):
        return "Error: provide a valid client email address."
    if not subject.strip() or not body.strip():
        return "Error: subject and body must not be empty."
    return _send_email(client_email, subject, body, "the client email has been sent")


def _send_email(to_email: str, subject: str, body: str, success_description: str) -> str:
    """Send an HTML email using the configured provider."""
    if os.getenv("USE_RESEND", "false").strip().lower() == "true":
        api_key = os.getenv("RESEND_API_KEY")
        from_email = os.getenv("RESEND_FROM_EMAIL")
        if not all([api_key, from_email]):
            return "Error: RESEND_API_KEY and RESEND_FROM_EMAIL must be configured."
        params: resend.Emails.SendParams = {
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": body,
        }
        resend.api_key = api_key
        try:
            email = resend.Emails.send(params)
            return f"Success: {success_description} (message ID: {email['id']})."
        except ResendError as error:
            message = str(error).replace(api_key, "[redacted]")[:500]
            return f"Error: Resend returned HTTP {error.code}: {message}"
        except Exception as error:
            message = str(error).replace(api_key, "[redacted]")[:500]
            return f"Error: could not send email through Resend ({message})."

    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    if not all([api_key, from_email]):
        return "Error: SENDGRID_API_KEY and SENDGRID_FROM_EMAIL must be configured."

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=body,
    )
    try:
        response = SendGridAPIClient(api_key).send(message)
        return f"Success: {success_description} (HTTP {response.status_code})."
    except Exception as error:
        return f"Error: could not send email through SendGrid ({error})."

if __name__ == "__main__":
    import asyncio

    # print(notify_sales_team("Company info here", "Matched services here"))
    # print(scrape_prospect_website("https://stscholastica.co.ke"))
    # print(notify_sales_team("Company info here", "Matched services here"))
    # print(get_our_services("Property management digital solutions.  Respond in 70 words only"))
    print(asyncio.run(search_services_and_projects("Property management digital solutions.  Respond in 70 words only")))
