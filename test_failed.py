"""Run only the currently failing cases — 5 API calls total."""
import requests, argparse

def chat(messages, url):
    r = requests.post(f"{url}/chat", json={"messages": messages}, timeout=35)
    return r.json()

def check(name, passed, detail=""):
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:7860")
args = parser.parse_args()
url = args.url.rstrip("/")
print(f"\nTesting 5 failing cases against {url}\n")

# 1. Off-topic Python function — must get 0 recs
resp = chat([{"role": "user", "content": "Write me a Python function to sort a list"}], url)
check("Off-topic Python function: no recs", len(resp.get("recommendations", [])) == 0, f"got {len(resp.get('recommendations', []))}")

# 2. Multi-turn Turn 1 — vague, must get 0 recs
msgs = [{"role": "user", "content": "We need a solution for senior leadership."}]
resp = chat(msgs, url)
check("Turn 1 vague: no recs", len(resp.get("recommendations", [])) == 0, f"got {len(resp.get('recommendations', []))}")
msgs.append({"role": "assistant", "content": resp["reply"]})

# 3. Multi-turn Turn 3 — must get recs after enough context
msgs.append({"role": "user", "content": "CXOs and directors, 15+ years experience."})
resp = chat(msgs, url)
msgs.append({"role": "assistant", "content": resp["reply"]})
msgs.append({"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."})
resp = chat(msgs, url)
check("Turn 3: has recs after context", len(resp.get("recommendations", [])) >= 1, f"got {len(resp.get('recommendations', []))}")
msgs.append({"role": "assistant", "content": resp["reply"]})

# 4. Multi-turn Turn 4 — refine must get recs
msgs.append({"role": "user", "content": "Can you also include cognitive ability tests?"})
resp = chat(msgs, url)
check("Turn 4 refine: has recs", len(resp.get("recommendations", [])) >= 1, f"got {len(resp.get('recommendations', []))}")

# 5. Compare — reply must be non-empty
resp = chat([{"role": "user", "content": "What's the difference between OPQ32r and Verify Numerical Reasoning?"}], url)
check("Compare: has reply text", len(resp.get("reply", "")) > 50, f"got {len(resp.get('reply', ''))} chars")
