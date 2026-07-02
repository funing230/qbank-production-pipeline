import os, sys, subprocess, json

# Keys from dad's message — 8 GPT API keys
keys = [
    "sk-B...KPVy",
    "sk-5...gnv4",
    "sk-f...tP1T",
    "sk-P...6irL",
    "sk-S...U2Lv",
    "sk-r...4MZq",
    "sk-V...9wzy",
    "sk-7...P6B7",
]
env_names = ["GPT5_API_KEY", "GPT_WORKER1_API_KEY", "GPT_WORKER2_API_KEY",
             "GPT_WORKER3_API_KEY", "GPT_WORKER4_API_KEY", "GPT_WORKER5_API_KEY",
             "GPT_WORKER6_API_KEY", "GPT_WORKER7_API_KEY"]

for name, key in zip(env_names, keys):
    os.environ[name] = key
os.environ["GPT_WORKER8_API_KEY"] = keys[7]

# Verify
for name in env_names:
    val = os.environ.get(name, "")
    assert len(val) > 40, f"{name} too short: {len(val)}"
print(f"All {len(env_names)} GPT keys set OK")

# Run production
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.exit(subprocess.run([
    sys.executable, "run_production_v7_queue.py",
    "--model", "gpt", "--enable-db-recycle"
]).returncode)
