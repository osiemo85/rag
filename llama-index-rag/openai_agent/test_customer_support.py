"""Small orchestration checks without model calls or outgoing email."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from agents import set_tracing_disabled
from openai_agent import customer_support as support

set_tracing_disabled(True)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_assessment_runs_planner_search_and_writer(self):
        state = support.ConversationState()
        calls = []

        async def fake_run(agent, prompt):
            calls.append(agent.name)
            if agent is support.conversation_agent:
                return SimpleNamespace(final_output=support.Decision(
                    action="assess", reply="", query="workflow automation", website_url=""
                ))
            if agent is support.planner_agent:
                return SimpleNamespace(final_output=support.SearchPlan(searches=[
                    support.SearchItem(reason="Find a service", query="automation")
                ]))
            if agent is support.search_agent:
                return SimpleNamespace(final_output="Verified automation service")
            return SimpleNamespace(final_output="We can help with automation. Does this fit?")

        with (
            patch.object(support, "trace", return_value=nullcontext()),
            patch.object(support.Runner, "run", side_effect=fake_run),
        ):
            answer = await support.respond("Can you automate our workflow?", state)

        self.assertIn("automation", answer)
        self.assertEqual(calls, [
            "Customer Support Agent", "Search Planner", "Service Search Agent",
            "Service Assessment Agent",
        ])
        self.assertEqual(len(state.history), 2)

    async def test_email_is_delegated_to_email_agent(self):
        state = support.ConversationState(history=[
            {"role": "assistant", "content": "We can help with automation."}
        ])
        calls = []

        async def fake_run(agent, prompt):
            calls.append(agent.name)
            if agent is support.conversation_agent:
                return SimpleNamespace(final_output=support.Decision(
                    action="email", reply="", query="", website_url=""
                ))
            self.assertIn("lead@example.com", prompt)
            return SimpleNamespace(final_output="Both emails sent.")

        with (
            patch.object(support, "trace", return_value=nullcontext()),
            patch.object(support.Runner, "run", side_effect=fake_run),
        ):
            answer = await support.respond("Send it to lead@example.com", state)

        self.assertEqual(answer, "Both emails sent.")
        self.assertEqual(calls, ["Customer Support Agent", "Email Agent"])
        self.assertEqual([tool.name for tool in support.email_agent.tools], [
            "send_email_to_agentrixx_team", "send_email_to_client"
        ])


if __name__ == "__main__":
    unittest.main()
