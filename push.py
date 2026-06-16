#!/usr/bin/env python3
"""Push git commits to GitHub via API (bypasses github.com connectivity issues)."""

import json, base64, subprocess, sys, os, urllib.request, ssl

REPO = "Xigorithm/vllm-ascend-workflow"
API = f"https://api.github.com/repos/{REPO}"
TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
ctx = ssl.create_default_context()

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

def api(endpoint, data=None, method=None):
    url = f"{API}{endpoint}"
    headers = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read())

if not TOKEN:
    print("Error: Set GH_TOKEN or GITHUB_TOKEN environment variable"); sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

local_sha = run("git rev-parse HEAD")
remote_ref = api("/git/refs/heads/main")
remote_sha = remote_ref["object"]["sha"]

if local_sha == remote_sha:
    print("Already up to date."); sys.exit(0)

commits = run(f"git rev-list {remote_sha}..HEAD").split("\n")
commits = [c for c in commits if c]
print(f"Pushing {len(commits)} commit(s)...")

for commit_sha in reversed(commits):
    msg = run(f"git log -1 --format=%B {commit_sha}")
    parent = run(f"git rev-parse {commit_sha}^")
    files = run(f"git diff-tree --no-commit-id -r --name-only {commit_sha}").split("\n")
    files = [f for f in files if f]

    parent_data = api(f"/git/commits/{parent}")
    base_tree = parent_data["tree"]["sha"]

    tree_entries = []
    for filepath in files:
        try:
            content = run(f"git show {commit_sha}:{filepath}")
            blob = api("/git/blobs", {"content": base64.b64encode(content.encode()).decode(), "encoding": "base64"})
            tree_entries.append({"path": filepath, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        except subprocess.CalledProcessError:
            pass

    new_tree = api("/git/trees", {"base_tree": base_tree, "tree": tree_entries})
    new_commit = api("/git/commits", {"message": msg, "tree": new_tree["sha"], "parents": [parent]})
    remote_sha = new_commit["sha"]
    print(f"  {commit_sha[:7]} -> {remote_sha[:7]}: {msg.split(chr(10))[0]}")

api("/git/refs/heads/main", {"sha": remote_sha}, method="PATCH")
try:
    run("git fetch origin main")
    run("git reset --hard origin/main")
except subprocess.CalledProcessError:
    print("Warning: git fetch failed. Push succeeded via API, local branch may be out of sync.")
print("Done!")