import os
import sys

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    try:
        response = client.models.generate_content(
            model=model,
            contents="Explique brievement la finance de marche.",
        )
    except Exception as exc:
        print(f"Gemini API call failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(response.text)


if __name__ == "__main__":
    main()
