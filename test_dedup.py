"""Self-check for save/update/history. Run: .venv/bin/python test_dedup.py (uses a temp DB)."""
import json
import os
import tempfile

os.environ["ENGRAM_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
import server  # noqa: E402

A = "KKK: マイページ編集ロックは発送6日前から。お届け日基準では DELIVERY_DAYS=3 なので9日前から変更不可。"
B = "KKK: 要件7の2回目表示は『お届け予定一覧』の2回目別受注のこと。マイページ詳細画面には出さない。"

a = server._save_memory(A, "p", "spec")
assert a["status"] == "created", a

# 同じ project の似た話題(旧実装では 0.90 以上で A を丸ごと上書きしていた)→ 新規
b = server._save_memory(B, "p", "spec")
assert b["status"] == "created" and b["id"] != a["id"], b

# ほぼ同一文 → 置換、旧版は history に残る
c = server._save_memory(A + " ", "p", "spec,decision")
assert c["status"] == "updated" and c["id"] == a["id"], c
h = json.loads(server.history(a["id"]))["versions"]
assert h[0]["content"] == A and h[0]["tags"] == "spec", h

# 自動保存の要約は手動の記憶と突き合わせない
d = server._save_memory(A, "p", "auto-save,sessionend")
assert d["status"] == "created", d
e = server._save_memory(A, "p", "auto-save,stop")
assert e["status"] == "updated" and e["id"] == d["id"], e

# 明示 update: tags 省略で維持、旧版が history に積まれる
u = json.loads(server.update(b["id"], "書き換え後"))
assert u["status"] == "updated", u
row = server._get_db().execute("SELECT content, tags FROM memories WHERE id = ?", [b["id"]]).fetchone()
assert (row["content"], row["tags"]) == ("書き換え後", "spec"), tuple(row)
assert json.loads(server.history(b["id"]))["versions"][0]["content"] == B
assert json.loads(server.update(99999, "x"))["status"] == "error"

print("ok")
