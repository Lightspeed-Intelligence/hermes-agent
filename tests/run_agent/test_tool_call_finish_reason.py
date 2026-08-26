"""Regression tests for invalid streamed tool-call finish-reason semantics."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _chunk(*, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=None,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="finish-reason-test", usage=None)


def _tool_delta(*, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=0, id=call_id, function=function)


def _stream_response(arguments: str, finish_reason: str):
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._fire_stream_delta = lambda _text: None

    chunks = [
        _chunk(tool_calls=[_tool_delta(call_id="call_1", name="terminal")]),
        _chunk(tool_calls=[_tool_delta(arguments=arguments)]),
        _chunk(finish_reason=finish_reason),
    ]
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda *args, **kwargs: iter(chunks)
    with (
        patch("run_agent.AIAgent._create_request_openai_client", return_value=client),
        patch("run_agent.AIAgent._close_request_openai_client"),
    ):
        return agent._interruptible_streaming_api_call({})


def test_repairable_arguments_keep_provider_finish_reason():
    response = _stream_response(
        '{"command": "ls -la", "timeout": 30',
        "tool_calls",
    )

    assert response.choices[0].finish_reason == "tool_calls"
    arguments = response.choices[0].message.tool_calls[0].function.arguments
    assert json.loads(arguments) == {"command": "ls -la", "timeout": 30}


def test_unrepairable_arguments_keep_provider_finish_reason_and_raw_payload():
    raw_arguments = '{"command": "unterminated'
    response = _stream_response(raw_arguments, "tool_calls")

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.arguments == raw_arguments


def test_unrepairable_arguments_keep_explicit_length_reason():
    response = _stream_response('{"command": "unterminated', "length")

    assert response.choices[0].finish_reason == "length"


@pytest.fixture()
def loop_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}
    return agent


def _tool_response(arguments: str, *, finish_reason: str = "tool_calls"):
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="terminal", arguments=arguments),
    )
    message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
        reasoning_content=None,
    )
    return SimpleNamespace(
        id="chatcmpl-invalid-tool-json",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )],
        usage=None,
    )


def _text_response(content: str):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
        reasoning_content=None,
    )
    return SimpleNamespace(
        id="chatcmpl-recovered",
        model="test/model",
        choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
        usage=None,
    )


def test_provider_completed_invalid_json_uses_bounded_recovery(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [
        _tool_response('{"command": "unterminated'),
        _text_response("Recovered after invalid tool arguments."),
    ]

    with (
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("run a command")

    assert loop_agent.client.chat.completions.create.call_count == 2
    assert result["completed"] is True
    assert result["final_response"] == "Recovered after invalid tool arguments."
