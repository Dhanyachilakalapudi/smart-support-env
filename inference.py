import os
import requests
from openai import OpenAI

# ✅ MUST use env variables (hackathon requirement)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

# ✅ fallback (local testing only)
if not API_BASE_URL:
    API_BASE_URL = "https://router.huggingface.co/v1"
if not API_KEY:
    API_KEY = "hf_dummy_key"

# ✅ correct OpenAI client (IMPORTANT)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

ENV_URL = "https://dhanyachilakalapudi-smart-support-env.hf.space"

def call_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("[ERROR] LLM call failed:", e)
        return None

def safe_request(method, url, **kwargs):
    try:
        r = requests.request(method, url, timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[ERROR] Request failed:", e)
        return None

def run_task(task_name):
    print(f"[START] task={task_name}")

    reset = safe_request("POST", f"{ENV_URL}/reset", json={})
    if not reset:
        print("Reset failed")
        return

    done = False
    step = 0

    while not done and step < 5:
        step += 1

        prompt = "Generate support response for login issue"

        llm_output = call_llm(prompt)

        action = {
            "workflow_step": "triage",
            "category": "authentication",
            "priority": "medium",
            "owner_team": "support_l1",
            "message_to_customer": llm_output if llm_output else "Please reset your password and try again.",
            "internal_note": "LLM generated"
        }

        result = safe_request(
            "POST",
            f"{ENV_URL}/step",
            json={"action": action}
        )

        if not result:
            break

        done = result.get("done", False)
        reward = result.get("reward")

        print(f"[STEP] step={step} reward={reward} done={done}")

    print(f"[END] task={task_name}\n")

def main():
    for task in ["easy", "medium", "hard"]:
        run_task(task)

if __name__ == "__main__":
    main()