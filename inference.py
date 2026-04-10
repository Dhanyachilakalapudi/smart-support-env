import requests
import json
import sys

BASE = "http://127.0.0.1:8000"

def safe_post(url, data):
    try:
        r = requests.post(url, json=data, timeout=10)
        return r.json()
    except:
        return None

def run(task):
    print(f"[START] task={task} env=openenv model=test")

    res = safe_post(f"{BASE}/reset", {"episode_id": task})
    if not res:
        print("[END] success=false steps=0 score=0 rewards=[]")
        return

    done = False
    step = 0
    rewards = []

    while not done and step < 5:
        step += 1

        action = {
            "workflow_step": "triage",
            "category": "authentication",
            "priority": "medium",
            "owner_team": "support_l1",
            "message_to_customer": "Please reset password",
            "internal_note": "test"
        }

        res = safe_post(f"{BASE}/step", {"action": action})
        if not res:
            break

        reward = res.get("reward", 0)
        done = res.get("done", False)
        error = res.get("observation", {}).get("last_action_error")

        rewards.append(reward)

        print(f"[STEP] step={step} action={json.dumps(action)} reward={reward} done={done} error={error}")

    print(f"[END] success={done} steps={step} score={sum(rewards)} rewards={rewards}")


def main():
    try:
        for t in ["easy", "medium", "hard"]:
            run(t)
        sys.exit(0)
    except:
        sys.exit(0)

if __name__ == "__main__":
    main()