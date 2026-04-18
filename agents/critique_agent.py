"""
agents/critique_agent.py

批判エージェント

DESIGN.md のエージェント設計:
  入力: 行動仮説リスト / 深掘り済み本文+原因分析(文脈として)
  出力: 各仮説への批判コメント

役割: 生成エージェントが出した仮説の弱点を、別エージェントとして客観的に指摘する。
      DESIGN.md「なぜ生成と批判を分けたか」の通り、同一エージェントに両方やらせると
      自分の出力に甘くなる偏りが生まれるため、構造的に分離している。

      批判の内容は再生成時に生成エージェントへそのまま渡されるため、
      「どの弱点をどう克服すべきか」が具体的に伝わる形で出力する。
"""

import json
import os
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# agents/ から親ディレクトリ(プロジェクトルート)の database.py を参照するためパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import add_criticism, get_hypotheses

load_dotenv()

# ─────────────────────────────────────────────
# システムプロンプト
# ─────────────────────────────────────────────

# 批判の観点を4つに絞り、明示的に指定している設計意図:
#
# 「批判してください」だけでは、モデルが「難しそう」「曖昧」など
# 表面的・否定的なコメントを返しがちになる。
# ユーザーが「再生成する」を選んだとき、批判が生成エージェントへそのまま渡される
# 設計になっているため、批判は「何をどう改善すべきか」が具体的に伝わる内容である必要がある。
#
# 4つの観点:
#   [実行可能性] 本当に今週中にできるか? 条件や前提に依存しすぎていないか?
#   [原因への効き目] 深掘りで判明した根本原因に対してこの行動は効くか?
#   [継続性] 一度やって終わりにならないか? 習慣として続く設計になっているか?
#   [副作用・リスク] この行動を取ることで別の問題が起きないか?
#
# この4観点を使うことで、「ただの否定」ではなく
# 「克服すべき弱点とその改善方向」を含む批判になる。
_SYSTEM_PROMPT = """\
あなたは行動仮説の批判専門エージェントです。

生成エージェントが提示した行動仮説を、以下の【批判の観点】から客観的に評価してください。
あなたの批判は後続の再生成プロセスで生成エージェントへそのまま渡されます。
そのため「どの弱点を、どう克服すべきか」が具体的に伝わる批判を書いてください。

## 批判の観点(各仮説に対して以下から最も重要な1〜2点を選んで指摘する)
- **[実行可能性]**: 今週中に本当に実行できるか? 他者・特定状況への依存が強すぎないか?
- **[原因への効き目]**: 分析で判明した根本原因(思考パターン・感情・信念)に対してこの仮説は効くか?
- **[継続性]**: 一度やって終わりにならないか? 意志力に頼りすぎていないか?
- **[副作用・リスク]**: この行動が別の問題や負担を生まないか?

## 批判の書き方ルール(「ただの否定」にならないための制約)
1. 弱点を指摘した後、必ず「〜することで改善できる」「〜を加えると実行しやすくなる」
   など、改善の方向性を一言添える
2. 「難しい」「曖昧」だけで終わらせない。何が難しいのか、どこが曖昧なのかを具体的に書く
3. 仮説を全否定しない。仮説の意図を尊重しながら「ここを変えると更に良くなる」という立場で書く

## 出力形式
必ず JSON のみを返してください。説明文・前置き・マークダウン記法は一切不要です。

{{
  "critiques": [
    "仮説1への批判コメント(100字以内)",
    "仮説2への批判コメント(100字以内)",
    "仮説3への批判コメント(100字以内)"
  ]
}}

critiques の件数は入力された仮説の件数と必ず一致させてください。
"""


# ─────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """API レスポンスから JSON を取り出す。
    generation_agent.py と同じフォールバック戦略を採用。
    解析に完全に失敗した場合は空リストを返してクラッシュを防ぐ。
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"critiques": []}


def _build_user_message(hypotheses: list[str], analyzed_text: str) -> str:
    """API に渡すユーザーメッセージを組み立てる。
    仮説リストと文脈(深掘り済み本文+原因分析)を1つのメッセージにまとめることで、
    エージェントが「この仮説は原因に対して本当に効くか」を判断できるようにしている。
    仮説に番号を振ることで、出力 critiques の順序と対応が取りやすくなる。
    """
    hypotheses_text = "\n".join(
        f"仮説{i + 1}: {h}" for i, h in enumerate(hypotheses)
    )
    return (
        f"【深掘り済み本文・原因分析(文脈として参照してください)】\n{analyzed_text}\n\n"
        f"【批判対象の行動仮説】\n{hypotheses_text}"
    )


# ─────────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────────

def run_critique(
    session_id: int,
    hypothesis_ids: list[int],
    hypotheses: list[str],
    analyzed_text: str,
) -> list[str]:
    """批判エージェントを実行して各仮説への批判コメントリストを返す。

    生成した批判は database.py の add_criticism() を使って自動保存する。
    hypothesis_ids と hypotheses の順序は必ず一致している前提で呼ぶこと。

    Args:
        session_id:       database.create_session() で得たセッションID。
                          現時点では直接使わないが、将来の拡張(批判のDB検索など)に備えて受け取る。
        hypothesis_ids:   database の hypotheses.id のリスト(保存先の紐付けに使う)
        hypotheses:       仮説テキストのリスト(hypothesis_ids と順序一致)
        analyzed_text:    深掘り済み本文 + 原因分析をまとめた文字列(文脈として渡す)

    Returns:
        批判コメントのリスト。hypotheses と同じ順序・同じ件数。
        API エラー時や件数不一致時は空リストを返す。
    """

    if not hypotheses:
        # 仮説が空の場合は早期リターン。批判する対象がないため。
        return []

    user_message = _build_user_message(hypotheses, analyzed_text)

    client = anthropic.Anthropic(api_key=os.getenv("API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    result = _parse_json_response(raw)
    critiques: list[str] = result.get("critiques", [])

    # 件数ガード: 仮説の件数と批判の件数が一致しない場合は保存しない。
    # 保存してしまうと hypothesis_ids との対応がずれてDBが壊れるため。
    if len(critiques) != len(hypotheses):
        return critiques

    # 各批判を対応する hypothesis_id に紐づけて保存する。
    # zip で hypothesis_id と批判テキストをペアにすることで順序の対応を保つ。
    for hyp_id, critique_text in zip(hypothesis_ids, critiques):
        add_criticism(hyp_id, critique_text)

    return critiques


# ─────────────────────────────────────────────
# 動作確認用（python -X utf8 agents/critique_agent.py で直接実行）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Windows のコンソールで日本語が文字化けしないよう UTF-8 に強制
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    from database import (
        add_hypothesis,
        create_session,
        get_criticisms,
        get_hypotheses,
        init_db,
        update_cause_analysis,
    )

    print("=== 批判エージェント 動作確認 ===\n")

    init_db()

    # テスト用セッションを作成
    session_id = create_session("今日のMTGで意見を言えなかった。", depth_level="normal")

    ANALYZED_TEXT = """\
【日記本文(深掘り済み)】
今日のチームMTGで自分の意見をまったく言えなかった。
発言しようとしたが、直前に「どうせ否定される」という思いが出てきて黙ってしまった。
以前に似た提案を一蹴された経験が頭から離れず、それ以来ずっと発言を避けてきた。

【原因分析】
表面的な原因は「発言を躊躇した」だが、背後には過去の失敗体験による回避パターンがある。
「提案=否定される」という信念が無意識に形成されており、自己防衛として沈黙を選んでいる。
"""

    update_cause_analysis(session_id, ANALYZED_TEXT)

    # 生成エージェントが出した仮説を想定してDBに登録
    SAMPLE_HYPOTHESES = [
        "[行動層] 次回MTGの最初の5分以内に「一言だけ感想を述べる」を実行し、小さな成功体験を積む",
        "[思考層] 「提案=否定される」を「提案=議論の材料を提供する行為」と再定義し、発言前に心の中でこのフレーズを唱える",
        "[環境層] MTG前日にSlackで簡潔な意見メモを共有するルーティンを作り、対面での発言への心理的ハードルを仕組みで下げる",
    ]

    hyp_ids = []
    for text in SAMPLE_HYPOTHESES:
        hid = add_hypothesis(session_id, text, generation=1)
        hyp_ids.append(hid)

    print("【批判対象の仮説】")
    for i, h in enumerate(SAMPLE_HYPOTHESES, 1):
        print(f"  [{i}] {h}")

    print("\n批判エージェントを実行中...\n")

    critiques = run_critique(session_id, hyp_ids, SAMPLE_HYPOTHESES, ANALYZED_TEXT)

    if critiques:
        print("【各仮説への批判】")
        for i, (hyp, crit) in enumerate(zip(SAMPLE_HYPOTHESES, critiques), 1):
            print(f"\n  仮説{i}: {hyp[:40]}...")
            print(f"  批判{i}: {crit}")
    else:
        print("※ 批判の生成に失敗しました")

    # DB に保存されたか確認
    print("\n" + "-" * 50)
    print("DB確認")
    for hid in hyp_ids:
        saved = get_criticisms(hid)
        print(f"  hypothesis_id={hid} の批判: {len(saved)}件")

    print("\n=== 動作確認完了 ===")
