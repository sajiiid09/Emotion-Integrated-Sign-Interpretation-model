import sys
import os
from brain.config import load_config
from brain.gemini_client import GeminiClient
from brain.intent import ResolvedIntent
from brain.prompt_builder import build_prompt

try:
    cfg = load_config()
    client = GeminiClient(cfg)
    print(f"GenAI Available: {client._available}")
    if client._client:
        print("Client initialized successfully.")
        
        resolved = ResolvedIntent(
            keywords=["কেমন", "আছো"],
            detected_emotion="neutral",
            resolved_emotion="neutral",
            meta={},
            flags={},
            notes=[],
            rule_trace=[]
        )
        prompt = build_prompt(resolved, cfg=cfg)
        
        print(f"Testing with model: {cfg.model_name}")
        text, meta = client.generate(prompt)
        print(f"Response: {text}")
        print(f"Meta: {meta}")
    else:
        print("Client NOT initialized (likely missing API key or SDK).")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()