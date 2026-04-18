"""
deep_dive_agent.py の動作確認サンプルコード

実行方法:
    python -X utf8 test_deep_dive.py

事前準備:
    .env に API_KEY=sk-... が設定されていること
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from database import init_db, create_session, get_session, get_drill_messages
from agents.deep_dive_agent import run_deep_dive

# ─────────────────────────────────────────────
# テスト用の日記と模擬回答
# ─────────────────────────────────────────────

DIARY = "今日の朝礼でチームへの改善提案を発言しようとしたが、結局言えなかった。"

# エージェントの質問に対してこの順番で答える（3回分）
MOCK_ANSWERS = [
    "上司に否定されそうで怖かったです。",
    "去年提案した時に「それは無理だ」と一言で切られました。",
    "その後チームの雰囲気が悪くなった気がして、発言が萎縮するようになりました。",
]

# ─────────────────────────────────────────────
# テスト実行
# ─────────────────────────────────────────────

print("=" * 50)
print("deep_dive_agent テスト")
print("=" * 50)
print(f"\n日記: {DIARY}\n")

init_db()
session_id = create_session(DIARY, depth_level="normal")
print(f"セッションID: {session_id}")
print("-" * 50)

# Turn 0: 最初の質問を生成（ユーザー回答なし）
result = run_deep_dive(session_id, DIARY)
turn = 0

while not result["done"]:
    print(f"\n[Turn {turn}] AI質問: {result['question']}")

    if turn < len(MOCK_ANSWERS):
        answer = MOCK_ANSWERS[turn]
        print(f"[Turn {turn}] ユーザー回答: {answer}")
    else:
        print("[Turn {turn}] 回答なし（想定外のターン数）")
        break

    result = run_deep_dive(session_id, DIARY, user_message=answer)
    turn += 1

# 完了
print("\n" + "=" * 50)
print("【原因分析】")
print("=" * 50)
print(result["cause_analysis"])

# DB 保存の確認
print("\n" + "-" * 50)
print("DB 確認")
print("-" * 50)
session = get_session(session_id)
msgs = get_drill_messages(session_id)

print(f"sessions.cause_analysis 保存: {'あり' if session['cause_analysis'] else 'なし'}")
print(f"drill_messages 件数: {len(msgs)} 件")
print()
for m in msgs:
    label = "AI  " if m["role"] == "assistant" else "User"
    print(f"  [{label}] {m['content'][:50]}...")
