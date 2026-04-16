"""
agents/generation_agent.py

生成エージェント

DESIGN.md のエージェント設計:
  入力: 深掘り済み本文+原因分析 / 仮説件数 / 前回の批判内容(再生成時のみ)
  出力: 行動仮説リスト(指定件数)

役割: 深掘り・分析エージェントが掘り起こした内容をもとに、
      「次にどんな行動をとるべきか」の仮説を複数提示する。
      再生成時は批判内容を受け取り、弱点を改善した仮説を出し直す。
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

from database import add_hypothesis, get_latest_generation

load_dotenv()

# ─────────────────────────────────────────────
# システムプロンプト
# ─────────────────────────────────────────────

# 仮説を「多様性のある3つの層」で生成させる設計意図:
#
# 単純に「3件の仮説を出して」と指示すると、
# 同じ方向性の似通った仮説(例: 「もっと練習する」「さらに練習する」「毎日練習する」)
# が出てしまう。これを防ぐために、各仮説を異なる「変化の層」から生成するよう誘導する。
#
# 層の設計:
#   [行動層] 今すぐ・具体的に変えられる「やること」。実行コストが低く即効性が期待できる。
#   [思考層] 認知・信念・判断基準を変える。行動が変わっても根本の思考が変わらないと再発するため。
#   [環境層] 仕組み・習慣・周囲の条件を変える。意志力に依存しない構造的な変化。
#
# この3層で分けることで、ユーザーは「行動面だけ」「思考面だけ」ではなく
# 多角的なアプローチを比較・選択できる。
_SYSTEM_PROMPT_BASE = """\
あなたは行動変化の仮説を生成する専門エージェントです。

ユーザーが書いた日記を深掘り・分析した結果をもとに、
「次に取るべき行動仮説」を {count} 件、以下の【層の制約】に従って生成してください。

## 層の制約(仮説の多様性を保つための設計)
各仮説は必ず異なる層から生成してください。似通った仮説は禁止です。

- **[行動層]**: 今週中に実行できる、具体的・小さな行動の変化
  例: 「○○という状況では、まず△△をやってみる」
- **[思考層]**: 自分の認知・信念・解釈パターンを変える試み
  例: 「○○という思い込みを、△△という視点で捉え直してみる」
- **[環境層]**: 意志力に頼らず、仕組み・習慣・条件を変える
  例: 「△△という状況を作ることで、自然と○○が起きやすくする」

## 各仮説に含めるべき要素
1. 何をするか(具体的な行動・変化の内容)
2. いつ・どんな場面でやるか(トリガーとなる状況)
3. なぜこれが原因分析に効くか(根拠の一言)

## 出力形式
必ず JSON のみを返してください。説明文・前置き・マークダウン記法は一切不要です。

{{"hypotheses": ["仮説1のテキスト", "仮説2のテキスト", "仮説3のテキスト"]}}

各仮説は150字以内の1文でまとめてください。
"""

# 再生成時に追加するプロンプト。
# 批判内容を明示的に渡すことで「批判された弱点を踏まえた改善版」を生成させる。
# 単に「改善してください」だけでは批判の焦点がぼけるので、批判テキストを丸ごと埋め込む。
_REGENERATION_SUFFIX = """\

## 重要: 再生成の指示
前回の仮説には以下の批判が寄せられました。
今回はこれらの弱点を克服した、より具体的・実行可能な仮説を生成してください。

【前回の批判】
{criticism}
"""


# ─────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """API レスポンスから JSON を取り出す。
    deep_dive_agent.py と同じフォールバック戦略を採用。
    Claude は稀に ```json ... ``` ブロックや余分なテキストを返すため、
    正規表現でも試みて、それでも失敗した場合は空リストを返す。
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # コードブロック内の JSON を試みる
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    # 最終フォールバック: 空リストで返してエラーを上位に伝える
    return {"hypotheses": []}


# ─────────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────────

def run_generation(
    session_id: int,
    analyzed_text: str,
    count: int = 3,
    previous_criticism: str = "",
) -> list[str]:
    """生成エージェントを実行して行動仮説リストを返す。

    生成した仮説は database.py の add_hypothesis() を使って自動保存する。
    再生成の場合は generation が自動的にインクリメントされる。

    Args:
        session_id:          database.create_session() で得たセッションID
        analyzed_text:       深掘り済み本文 + 原因分析をまとめた文字列
                             (呼び出し側が「日記本文\n\n原因分析: ...」の形で組み立てて渡す)
        count:               生成する仮説件数。Phase 1 では 3 固定。
        previous_criticism:  再生成時のみ渡す。前回の批判内容(文字列)。
                             初回は空文字列のままでよい。

    Returns:
        仮説テキストのリスト。例: ["仮説1のテキスト", "仮説2のテキスト", ...]
        API エラーや JSON 解析失敗時は空リストを返す。
    """

    # システムプロンプトに仮説件数を埋め込む
    # count を埋め込むことで、将来 Phase 2 で件数選択に対応したとき
    # プロンプトを書き換えずにそのまま使える
    system = _SYSTEM_PROMPT_BASE.format(count=count)

    # 再生成時: 批判内容を追記してプロンプトを強化する
    # 空文字列チェックをして初回は追記しない(余分な指示でモデルを混乱させないため)
    if previous_criticism.strip():
        system += _REGENERATION_SUFFIX.format(criticism=previous_criticism.strip())

    # ユーザーメッセージとして渡す内容を組み立てる。
    # 「深掘り済み本文」と「原因分析」を1つの文字列にまとめて渡すことで、
    # エージェントが文脈と分析結果を一度に参照できる。
    user_message = f"【深掘り済み本文・原因分析】\n{analyzed_text}"

    client = anthropic.Anthropic(api_key=os.getenv("API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    result = _parse_json_response(raw)
    hypotheses: list[str] = result.get("hypotheses", [])

    # DB への保存
    # get_latest_generation() で現在の最大世代番号を取得し、+1 して今回の generation とする。
    # 初回は 0 が返るので generation=1 になる。再生成ごとに 2, 3, ... とインクリメントされる。
    current_generation = get_latest_generation(session_id) + 1
    for hyp_text in hypotheses:
        add_hypothesis(session_id, hyp_text, generation=current_generation)

    return hypotheses


# ─────────────────────────────────────────────
# 動作確認用（python -X utf8 agents/generation_agent.py で直接実行）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Windows のコンソールで日本語が文字化けしないよう UTF-8 に強制
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    from database import create_session, get_hypotheses, get_latest_generation, init_db

    print("=== 生成エージェント 動作確認 ===")
    print("(終了するには Ctrl+C)\n")

    init_db()

    # テスト用の深掘り済みテキスト(実際は deep_dive_agent の出力が入る)
    SAMPLE_ANALYZED_TEXT = """\
【日記本文（深掘り済み）】
今日のチームMTGで自分の意見をまったく言えなかった。
提案しようとしたが、発言しようとする直前に「どうせ否定される」という思いが出てきて黙ってしまった。
以前に似た提案を一蹴された経験が頭から離れず、それ以来ずっと発言を避けてきた。

【原因分析】
表面的な原因は「発言を躊躇した」だが、背後には過去の失敗体験による回避パターンがある。
「提案=否定される」という信念が無意識に形成されており、自己防衛として沈黙を選んでいる。
この思考パターンは「失敗を避けることで安全を保つ」という信念から来ており、
長期的にはチームへの貢献機会を失い、自己評価の低下につながるリスクがある。
"""

    print("【テスト1】初回生成(批判なし)\n")
    # テスト用セッションを作成
    session_id = create_session("テスト: チームMTGで意見を言えなかった", depth_level="normal")
    print(f"セッションID: {session_id}")

    hypotheses = run_generation(session_id, SAMPLE_ANALYZED_TEXT, count=3)

    if hypotheses:
        print(f"\n生成された仮説 ({len(hypotheses)}件):")
        for i, h in enumerate(hypotheses, 1):
            print(f"  [{i}] {h}")
    else:
        print("※ 仮説の生成に失敗しました")

    # DB に保存されたか確認
    saved = get_hypotheses(session_id, generation=1)
    print(f"\nDB確認 - generation=1 の仮説数: {len(saved)}件")
    gen = get_latest_generation(session_id)
    print(f"DB確認 - latest_generation: {gen}")

    print("\n" + "=" * 50)
    print("【テスト2】再生成(批判あり)\n")

    SAMPLE_CRITICISM = """\
- 仮説1: 「次のMTGで意見を1つ言う」は目標が曖昧で、どんな状況でも実行できるか不明。
- 仮説2: 「信頼できる同僚に事前に話す」は相手依存で、相手が不在の場合に実行できない。
- 仮説3: 「発言前に3秒考える」は表面的な対処で、根本の回避パターンへのアプローチが弱い。
"""

    hypotheses2 = run_generation(
        session_id,
        SAMPLE_ANALYZED_TEXT,
        count=3,
        previous_criticism=SAMPLE_CRITICISM,
    )

    if hypotheses2:
        print(f"再生成された仮説 ({len(hypotheses2)}件):")
        for i, h in enumerate(hypotheses2, 1):
            print(f"  [{i}] {h}")
    else:
        print("※ 仮説の再生成に失敗しました")

    # DB に保存されたか確認(generation=2 になっているはず)
    saved2 = get_hypotheses(session_id, generation=2)
    print(f"\nDB確認 - generation=2 の仮説数: {len(saved2)}件")
    gen2 = get_latest_generation(session_id)
    print(f"DB確認 - latest_generation: {gen2}")

    print("\n=== 動作確認完了 ===")
