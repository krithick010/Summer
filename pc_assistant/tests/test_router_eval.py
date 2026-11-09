import json
import httpx

tools = [
    {
        "name": "system_info",
        "description": "Returns CPU, RAM, and Disk telemetry for this computer.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "send_email",
        "description": "Sends an email to a specific email address.",
        "parameters": {
            "type": "object",
            "properties": {
                "to_email": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to_email", "subject", "body"]
        }
    }
]

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intent classifier for a PC assistant.
Your job is to classify the user's message into one of three actions:
1. "system_info" - ONLY if the user specifically asks to check system/PC status, CPU, RAM, memory, hardware, or disk.
2. "send_email" - ONLY if the user explicitly asks to email someone or send an email.
3. "none" - for any greeting, small talk, question, or casual conversation ("hi", "wassup", "how are you", "who are you").

Respond strictly with valid JSON:
{{"plugin_name": "system_info" | "send_email" | "none", "kwargs": {{}}}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def test_intent(msg: str):
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt_template.format(user_msg=msg),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }
    resp = httpx.post("http://localhost:11434/api/generate", json=payload, timeout=30.0)
    return resp.json().get("response")

test_cases = [
    "Wassup bro ...how are you?",
    "Hello there",
    "How much RAM is free?",
    "Check system stats",
    "Send an email to alex@test.com saying hello",
    "Tell me a joke"
]

for tc in test_cases:
    print(f"{tc:50} -> {test_intent(tc)}")

