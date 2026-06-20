from __future__ import annotations

import os


class GeminiPlannerClient:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.8, "response_mime_type": "application/json"},
            request_options={"timeout": 60},
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

