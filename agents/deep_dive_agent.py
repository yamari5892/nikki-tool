"""
agents/deep_dive_agent.py

深掘り・分析エージェント

DESIGN.md のエージェント設計:
  入力: 日記本文 / 深掘りレベル / 対話履歴
  出力: 次の質問(done=False) または 完了フラグ+原因分析(done=True)

役割: 対話で情報を引き出しながら、最終的に原因分析も行う統合エージェント。
DESIGN.md に「なぜ深掘りと分析を統合したか」とある通り、
情報が十分かどうかの判断も同じエージェントに委ねることで、
柔軟かつ少ないターン数で高品質な分析が得られる。
"""

import json
import os
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# agents/ から親ディレクトリ(プロジェクトルート)の database.py を参照するためパスを追加
# __file__ の親の親 = プロジェクトルート
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import add_drill_message, get_drill_messages, update_cause_analysis

load_dotenv()

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────

# 深掘りレベルごとの最大ユーザー応答回数
# shallow=0: 対話なしで即座に原因分析へ（user_turns=0 >= 0 で force_finish が立つ）
# deep=6: エージェントが十分と判断した場合はより早く終わることもある
MAX_USER_TURNS: dict[str, int] = {
    "shallow": 0,
    "normal": 3,
    "deep": 6,
}

# ─────────────────────────────────────────────
# システムプロンプト
# ─────────────────────────────────────────────

# {max_turns} は実行時に .format() で埋める
_SYSTEM_PROMPT_TEMPLATE = """\
あなたは日記の深掘り・原因分析エージェントです。

ユーザーが書いた日記の内容をもとに、対話を通じて出来事の背景・感情・思考を深く掘り下げ、
最終的にその根本原因を分析することがあなたの役割です。

## 質問のルール
- 1回の返答では必ず1つだけ質問する(複数同時に聞かない)
- ユーザーの答えの中で最も「なぜ?」を掘り下げるべき部分を選んで質問する
- 「その時どう感じましたか」「なぜそう判断しましたか」「具体的にどんな状況でしたか」など、
  行動・判断の背後にある感情・動機・信念を引き出す質問を優先する
- 表面的な事実確認ではなく、思考パターンや無意識の前提を掘り起こすことを意識する

## 終了の判断
あなたには最大 {max_turns} 回の質問機会があります。
以下のいずれかを満たしたら深掘りを終了して原因分析に移ってください:
- 質問機会を使い切った
- 出来事の背景・感情・思考が十分に掘り下げられ、原因分析に必要な情報が揃った
  (残り質問機会があっても終了してよい)

## 出力形式
必ず JSON のみを返してください。説明文や前置きは一切不要です。

深掘り継続の場合:
{{"done": false, "question": "次の質問テキスト"}}

深掘り完了の場合:
{{"done": true, "cause_analysis": "原因分析テキスト(200〜400字)"}}

原因分析では以下を含めてください:
- 出来事の表面的な原因
- その背後にある思考パターン・感情・信念
- なぜそのパターンが生まれたか(対話から読み取れる範囲で)
"""

# ─────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────

def _count_user_turns(messages: list[dict]) -> int:
    """対話履歴の中でユーザーが発言した回数を数える。
    深掘り終了タイミングの判定に使う。
    """
    return sum(1 for m in messages if m["role"] == "user")


def _parse_json_response(raw: str) -> dict:
    """API レスポンスから JSON を取り出す。
    Claude は通常 JSON のみを返すが、稀に ```json ... ``` のコードブロックや
    前後に余分なテキストが入ることがある。正規表現でフォールバック抽出する。
    それでも取れない場合は安全側(done=True)に倒してそのまま原因分析として扱う。
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
        # 最終フォールバック: テキストをそのまま原因分析として返す
        return {"done": True, "cause_analysis": raw}


# ─────────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────────

def run_deep_dive(
    session_id: int,
    diary_text: str,
    user_message: str | None = None,
    depth_level: str = "normal",
) -> dict:
    """深掘り・分析エージェントを1ターン進める。

    呼び出し側は「done=False なら question を表示してユーザーに答えさせ、
    done=True なら cause_analysis を表示して次のステップへ」という
    シンプルなループを組むだけでよい。

    Args:
        session_id:   database.create_session() で得たセッションID
        diary_text:   元の日記本文(毎回渡すことで文脈を保持)
        user_message: ユーザーの最新の回答。最初の呼び出し(最初の質問を出すだけ)は None
        depth_level:  深掘りレベル。Phase 1 では "normal" 固定

    Returns:
        {"done": False, "question": "次の質問テキスト"}
        または
        {"done": True, "cause_analysis": "原因分析テキスト"}
    """

    # ユーザーの返答を先に DB に保存する
    # こうすることで次の get_drill_messages() が最新の履歴を返す
    if user_message is not None:
        add_drill_message(session_id, "user", user_message)

    # DB から対話履歴を取得
    # セッションが中断・再開しても正確に復元されるよう、毎回 DB から読む
    history = get_drill_messages(session_id)
    user_turns = _count_user_turns(history)
    max_turns = MAX_USER_TURNS.get(depth_level, MAX_USER_TURNS["normal"])

    # ユーザーの応答回数が上限に達していたら強制終了フラグを立てる
    # エージェント任せだけだと稀に続けようとするため、呼び出し側でも確実に管理する
    force_finish = user_turns >= max_turns

    # システムプロンプトに最大ターン数を埋め込む
    system = _SYSTEM_PROMPT_TEMPLATE.format(max_turns=max_turns)

    # 強制終了の場合は追加指示でエージェントに必ず done=true を出力させる
    if force_finish:
        system += (
            "\n\n## 重要\n"
            "これは最後の返答です。必ず done=true の JSON で原因分析を出力してください。"
        )

    # Anthropic API に渡す messages を組み立てる
    # 日記本文を最初のユーザーメッセージとして渡すことで、
    # エージェントが常に元の文脈を参照できるようにしている
    # その後に DB から取得した対話履歴を続ける
    api_messages = [{"role": "user", "content": f"【日記本文】\n{diary_text}"}] + history

    client = anthropic.Anthropic(api_key=os.getenv("API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system,
        messages=api_messages,
    )

    raw = response.content[0].text.strip()
    result = _parse_json_response(raw)

    # エージェントの返答を DB に保存
    # done=True の場合は cause_analysis を保存し、sessions テーブルにも書き込む
    if result.get("done"):
        agent_text = result.get("cause_analysis", "")
        # sessions.cause_analysis に書き込むことで、後続の生成エージェントが参照できる
        update_cause_analysis(session_id, agent_text)
    else:
        agent_text = result.get("question", "")

    add_drill_message(session_id, "assistant", agent_text)

    return result


# ─────────────────────────────────────────────
# 動作確認用（python -X utf8 agents/deep_dive_agent.py で直接実行）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Windows のコンソールで日本語が文字化けしないよう UTF-8 に強制
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    # database.py の初期化もここで行う
    from database import create_session, get_session, init_db

    print("=== 深掘り・分析エージェント 動作確認 ===")
    print("(終了するには Ctrl+C)\n")

    init_db()

    diary = input("【日記本文を入力してください】\n> ")
    session_id = create_session(diary, depth_level="normal")
    print(f"\nセッションID: {session_id}\n")

    # 最初の呼び出し: user_message=None で最初の質問を生成させる
    result = run_deep_dive(session_id, diary)

    # done=True になるまで対話を繰り返す
    while not result.get("done"):
        print(f"\nAI: {result['question']}")
        answer = input("あなた: ")
        result = run_deep_dive(session_id, diary, user_message=answer)

    print("\n" + "=" * 40)
    print("【原因分析】")
    print(result["cause_analysis"])
    print("=" * 40)

    # DB への保存確認
    session = get_session(session_id)
    saved = "あり" if session and session["cause_analysis"] else "なし"
    print(f"\nDB確認 - cause_analysis 保存: {saved}")

    from database import get_drill_messages
    msgs = get_drill_messages(session_id)
    print(f"DB確認 - drill_messages 件数: {len(msgs)}件")
