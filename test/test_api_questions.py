import os
import json
import requests
from datetime import datetime

# Configurable API endpoint
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
INPUT_JSON = os.path.join(os.path.dirname(__file__), "diabetes_questions.json")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "test_results.json")

# Optionally, set a session_id for all questions (or use a new one per run)
SESSION_ID = os.getenv("TEST_SESSION_ID")


def load_questions():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accepts either a list of questions or a dict with a 'questions' key
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data


def query_api(question, session_id=None):
    payload = {"query": question}
    if session_id:
        payload["session_id"] = session_id
    try:
        resp = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    questions = load_questions()
    results = []
    sid = SESSION_ID or f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Querying: {q}")
        result = query_api(q, session_id=sid)
        results.append({
            "question": q,
            "result": result
        })
    # Save results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "backend_url": BACKEND_URL,
            "session_id": sid,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
