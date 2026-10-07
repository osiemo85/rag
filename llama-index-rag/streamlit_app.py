"""Streamlit UI for the customer support agent.

Run with:
    uv run streamlit run streamlit_app.py
"""

import asyncio

import streamlit as st
from llama_index.core.workflow import Context

from customer_support_agent import agent


st.set_page_config(
    page_title="Customer Support Agent",
    page_icon="🤖",
    layout="centered",
)


def reset_conversation() -> None:
    """Start a fresh agent conversation and clear the visible messages."""
    st.session_state.agent_context = Context(agent)
    st.session_state.messages = []


def get_agent_context() -> Context:
    """Create the agent context once for this browser session."""
    if "agent_context" not in st.session_state:
        st.session_state.agent_context = Context(agent)
    return st.session_state.agent_context


async def ask_agent(user_message: str) -> str:
    """Send a message to the agent using the current session context."""
    response = await agent.run(
        user_msg=user_message,
        ctx=get_agent_context(),
    )
    return str(response)


def main() -> None:
    st.title("🤖 Customer Support Agent")
    st.caption("Tell us about your business and discover how our services can help.")

    with st.sidebar:
        st.header("Conversation")
        st.button("Clear chat", on_click=reset_conversation, use_container_width=True)
        st.info(
            "The agent may ask for your website and email so it can assess your needs "
            "and email you and the sales team when there is a strong match."
        )

    # Re-render the existing conversation after every Streamlit rerun.
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("How can we help your business?")
    if not user_message:
        return

    st.session_state.setdefault("messages", []).append(
        {"role": "user", "content": user_message}
    )
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = asyncio.run(ask_agent(user_message))
            except Exception as error:
                response = f"Sorry, I couldn't process that request: {error}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
