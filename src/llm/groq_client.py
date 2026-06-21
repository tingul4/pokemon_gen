from __future__ import annotations

import os


class GroqPlannerClient:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty response.")
        return text
