# engram

Long-term memory MCP server for Claude Code — hybrid search (SQLite FTS5 + sqlite-vec) with time decay and auto-pruning.

セッションをまたいで永続化する記憶を Claude Code に提供する MCP サーバーです。キーワード検索とベクトル検索（日本語対応）を組み合わせ、時間経過で重要度が減衰する仕組みを備えています。

## 特徴

- **ハイブリッド検索**: SQLite FTS5 (trigram) によるキーワード検索 + sqlite-vec によるベクトル検索を **RRF (Reciprocal Rank Fusion)** で統合
- **日本語対応の埋め込み**: [Ruri v3 310m](https://huggingface.co/cl-nagoya/ruri-v3-310m) を使用
- **時間減衰**: 半減期 30 日の指数減衰スコアリング
- **重複排除**: 同じ project でコサイン類似度 0.98 以上(ほぼ同一文)の記憶だけ置き換える。似ているだけの記憶は置き換えず `similar` として返す。自動保存(`auto-save` タグ)の要約どうしは 0.90 以上で統合し、手動の記憶とは突き合わせない
- **上書き履歴**: 置き換え前の本文とタグは `memories_history` に残り、`history(memory_id)` で見られる
- **自動プルーニング**: 最大 10,000 件を超えると、ヒット数の少ない古いものから削除
- **プロジェクト別管理**: `project` フィールドでスコープを分離可能

## 必要環境

- Python 3.10 以上（開発は 3.12 で確認）
- macOS / Linux
- 初回起動時に Hugging Face から埋め込みモデル（約 600MB）をダウンロードします

## インストール

```bash
git clone https://github.com/KimMaru10/engram.git ~/.claude/engram
cd ~/.claude/engram

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Claude Code への登録

`~/.claude.json` の `mcpServers` に以下を追加します（パスはクローン先に合わせて変更してください）。

```json
{
  "mcpServers": {
    "engram": {
      "type": "stdio",
      "command": "/Users/<you>/.claude/engram/.venv/bin/python",
      "args": ["/Users/<you>/.claude/engram/server.py"],
      "env": {}
    }
  }
}
```

または CLI で:

```bash
claude mcp add engram \
  /Users/<you>/.claude/engram/.venv/bin/python \
  /Users/<you>/.claude/engram/server.py
```

登録後、Claude Code を再起動すると engram のツールが利用可能になります。

## 提供する MCP ツール

| ツール | 説明 |
| --- | --- |
| `save(content, project="", tags="")` | 記憶を新規保存。ほぼ同一の記憶だけ置き換える。似た記憶は結果の `similar` に id と類似度を返す |
| `update(memory_id, content, tags=None)` | 指定 ID の記憶を意図して置き換える(tags 省略で維持)。旧版は履歴に残る |
| `history(memory_id)` | 指定 ID の過去の版を新しい順に表示 |
| `search(query, project="", limit=5)` | ハイブリッド検索（キーワード + 意味）+ 時間減衰でランキング |
| `prune(older_than_days=90, project="")` | 指定日数アクセスのない記憶を削除 |
| `stats(project="")` | 件数・最古/最新・プロジェクト別統計・DB サイズ |
| `delete(memory_id)` | 指定 ID の記憶を削除 |

## 設定（環境変数）

| 変数 | デフォルト | 説明 |
| --- | --- | --- |
| `ENGRAM_DB_PATH` | `~/.claude/engram/memory.db` | SQLite DB の保存先 |

`server.py` 上部の定数で次のパラメータも調整できます。

| 定数 | デフォルト | 説明 |
| --- | --- | --- |
| `MODEL_NAME` | `cl-nagoya/ruri-v3-310m` | 埋め込みモデル |
| `HALF_LIFE_DAYS` | `30` | 時間減衰の半減期（日） |
| `MAX_MEMORIES` | `10000` | 上限件数（超過分は古い順に自動削除） |
| `DEDUP_THRESHOLD` | `0.98` | 手動保存で置き換えるしきい値(ruri-v3 では無関係な同 project の記憶同士でも中央値 0.87 前後になるため高めにする) |
| `SIMILAR_THRESHOLD` | `0.90` | `similar` として返すしきい値 |
| `AUTO_SAVE_DEDUP_THRESHOLD` | `0.90` | `auto-save` タグの要約どうしを統合するしきい値 |
| `RRF_K` | `60` | RRF 定数 |

## アーキテクチャ

```
┌──────────────┐
│ Claude Code  │
└──────┬───────┘
       │ stdio (MCP)
┌──────▼───────────────────────────────────┐
│            engram (server.py)            │
├──────────────────────────────────────────┤
│  save / update / history / search /      │
│  prune / stats / delete                  │
└──────┬───────────────────────────────────┘
       │
┌──────▼─────────────────────────────────┐
│ SQLite (memory.db)                     │
│  ├─ memories         (本体)            │
│  ├─ memories_fts     (FTS5 trigram)    │
│  └─ memories_vec     (sqlite-vec, 768) │
└────────────────────────────────────────┘
```

検索時は FTS5 と vec0 をそれぞれランキングし、RRF でスコア統合 → 各記憶の作成日からの時間減衰係数を掛けて最終順位を決定します。

## ライセンス

MIT
