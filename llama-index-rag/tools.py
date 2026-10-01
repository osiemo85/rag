"""Tools used by the customer-support agent."""

import html
import os
import re
from functools import lru_cache
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.cerebras import Cerebras
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
    """Search Agentrixx's services knowledge base for relevant information.

    Args:
        prompt (str): A prospect's question, business need, or pain point.

    Returns:
        str: Relevant service information or a readable search error.
    """
    if not prompt.strip():
        return "Please provide a question about our services."

    try:
        query_engine = _service_index().as_query_engine()
        return str(query_engine.query(prompt))
    except Exception as error:
        return f"Unable to search the services knowledge base: {error}"



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
    """Send a qualified-lead email to the configured sales inbox through SendGrid.

    Args:
        subject (str): A concise subject describing the qualified lead.
        body (str): Lead contact details, business information, needs, and
            matched Agentrixx services.

    Returns:
        str: A delivery confirmation or a readable configuration or delivery error.
    """
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    to_email = os.getenv("TO_EMAIL")
    if not all([api_key, from_email, to_email]):
        return "Error: SENDGRID_API_KEY, SENDGRID_FROM_EMAIL, and TO_EMAIL must be configured."

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=body,
    )
    try:
        response = SendGridAPIClient(api_key).send(message)
        return f"Success: the sales team has been notified (HTTP {response.status_code})."
    except Exception as error:
        return f"Error: could not send email through SendGrid ({error})."

if __name__ == "__main__":
    # print(notify_sales_team("Company info here", "Matched services here"))
    print(scrape_prospect_website("https://citizen.digital"))
