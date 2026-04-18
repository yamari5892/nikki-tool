"""
app_v2.py

Phase 1 統合UI

DESIGN.md のユーザー操作フローに従った5画面構成:
  1. input    : 日記入力 + 深掘りレベル選択
  2. deep_dive: 深掘り対話(AIと往復)
  3. analysis : 原因分析表示 + 「仮説を生成」
  4. result   : 仮説 + 批判の表示 + 「確定」or「再生成」
  5. done     : 保存完了

画面遷移は st.session_state.page を唯一の真実として管理する。
状態変更後は必ず st.rerun() で再描画をトリガーする。
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# app_v2.py はプロジェクトルートに置くので、
# agents/ と database.py はそのまま import できる
sys.path.insert(0, str(Path(__file__).parent))

from agents.critique_agent import run_critique
from agents.deep_dive_agent import run_deep_dive
from agents.generation_agent import run_generation
from database import (
    complete_session,
    create_session,
    get_drill_messages,
    get_hypotheses,
    init_db,
    set_final_hypotheses,
    update_cause_analysis,
)

load_dotenv()

# ─────────────────────────────────────────────
# 初期化
# ─────────────────────────────────────────────

# アプリ起動時に必ずテーブルを作成する。IF NOT EXISTS なので副作用なし。
init_db()

st.set_page_config(page_title="nikki-tool", page_icon="📓", layout="centered")


def _init_session_state():
    """session_state のキーをまとめて初期化する。
    キーが既に存在する場合は上書きしないので、リロードしても状態が消えない。
    """
    defaults = {
        "page": "input",          # 現在の画面
        "session_id": None,       # DB セッション ID
        "diary_text": "",         # ユーザーが書いた日記
        "current_question": "",   # 深掘りエージェントからの現在の質問
        "cause_analysis": "",     # 原因分析テキスト
        "analyzed_text": "",      # 日記 + 原因分析をまとめた文字列(生成・批判エージェントへ渡す)
        "hypotheses": [],         # 現在の世代の仮説テキストリスト
        "hyp_ids": [],            # 現在の世代の仮説 DB ID リスト
        "critiques": [],          # 各仮説への批判テキストリスト
        "regen_count": 0,         # 再生成の回数。上限を超えたらボタンを無効化してAPI費用の暴走を防ぐ
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


def _go(page: str):
    """画面遷移ヘルパー。page を更新して即座に再描画する。"""
    st.session_state.page = page
    st.rerun()


def _reset():
    """全状態をリセットして入力画面に戻る。"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ─────────────────────────────────────────────
# 画面1: 入力画面
# ─────────────────────────────────────────────

def page_input():
    st.title("📓 nikki-tool")
    st.caption("日記を書いて、次の行動仮説を見つけよう")
    st.divider()

    diary = st.text_area(
        "今日の日記",
        placeholder="今日あったこと、感じたこと、考えたことを自由に書いてください",
        height=180,
    )

    # Phase 2 で3段階選択を実装するとき、ここをセレクトボックスに変えるだけで動く
    st.caption("深掘りレベル: 普通（Phase 1 固定）")

    if st.button("送信して深掘りを始める", type="primary", disabled=not diary.strip()):
        with st.spinner("セッションを準備中..."):
            # DB にセッションを作成して以降の書き込みの基点にする
            sid = create_session(diary.strip(), depth_level="normal")
            st.session_state.session_id = sid
            st.session_state.diary_text = diary.strip()

            # 最初の質問を生成する(user_message=None = 最初の呼び出し)
            result = run_deep_dive(sid, diary.strip())

        if result.get("done"):
            # レベルが shallow の場合など、即完了するケースへの対応
            _handle_deep_dive_done(result["cause_analysis"])
        else:
            st.session_state.current_question = result["question"]
            _go("deep_dive")


# ─────────────────────────────────────────────
# 画面2: 深掘り対話画面
# ─────────────────────────────────────────────

def page_deep_dive():
    st.title("深掘り対話")
    st.caption("AIの質問に答えることで、出来事の背景を掘り下げます")
    st.divider()

    # DB から対話履歴を取得して表示する。
    # session_state に会話履歴を持たずに DB から読む設計にしているのは、
    # ブラウザリロード後も正確に復元できるようにするため。
    history = get_drill_messages(st.session_state.session_id)

    # AIの最初のメッセージ(日記本文への最初の質問)だけは履歴に含まれないので
    # current_question を先頭に表示する。それ以降は履歴から表示する。
    # ただし履歴にすでに assistant の発言が入っていれば current_question は重複するため、
    # 履歴が空(=最初のターン)のときだけ current_question を単独で出す。
    if not history:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.current_question)
    else:
        # 過去の対話履歴をすべて表示
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 最後が assistant 発言でなければ current_question を表示
        # (ユーザーが回答した直後のリロードで二重表示しないためのガード)
        if history[-1]["role"] == "user":
            with st.chat_message("assistant"):
                st.markdown(st.session_state.current_question)

    # ユーザーの回答入力欄
    answer = st.chat_input("答えを入力してください...")
    if answer:
        with st.chat_message("user"):
            st.markdown(answer)

        with st.chat_message("assistant"):
            with st.spinner("考えています..."):
                result = run_deep_dive(
                    st.session_state.session_id,
                    st.session_state.diary_text,
                    user_message=answer,
                )

        if result.get("done"):
            _handle_deep_dive_done(result["cause_analysis"])
        else:
            st.session_state.current_question = result["question"]
            st.rerun()


def _handle_deep_dive_done(cause_analysis: str):
    """深掘り完了時の共通処理。
    原因分析を保存し、生成・批判エージェントへ渡す analyzed_text を組み立てて
    分析表示画面へ遷移する。
    """
    st.session_state.cause_analysis = cause_analysis

    # 生成エージェントと批判エージェントに渡す文字列を組み立てる。
    # 日記本文と原因分析を1つの文字列にまとめることで、
    # 両エージェントが文脈と分析結果を一度に参照できる。
    st.session_state.analyzed_text = (
        f"【日記本文】\n{st.session_state.diary_text}\n\n"
        f"【原因分析】\n{cause_analysis}"
    )
    _go("analysis")


# ─────────────────────────────────────────────
# 画面3: 原因分析表示画面
# ─────────────────────────────────────────────

def page_analysis():
    st.title("原因分析")
    st.caption("深掘りの結果から、行動のパターンと根本原因を分析しました")
    st.divider()

    st.subheader("分析結果")
    st.info(st.session_state.cause_analysis)

    st.divider()

    # Phase 2 で件数選択を実装するとき、ここをスライダーに変えるだけで動く
    st.caption("仮説件数: 3件（Phase 1 固定）")

    if st.button("仮説を生成する", type="primary"):
        with st.spinner("行動仮説を生成中..."):
            hypotheses = run_generation(
                st.session_state.session_id,
                st.session_state.analyzed_text,
                count=3,
            )

        if not hypotheses:
            st.error("仮説の生成に失敗しました。もう一度お試しください。")
            return

        # 生成した仮説を DB から取得して hypothesis_id を得る。
        # run_generation の内部で add_hypothesis が呼ばれているため、
        # ここでは get_hypotheses で最新世代を取り出すだけでよい。
        from database import get_hypotheses, get_latest_generation
        gen = get_latest_generation(st.session_state.session_id)
        saved = get_hypotheses(st.session_state.session_id, generation=gen)

        st.session_state.hypotheses = [h["content"] for h in saved]
        st.session_state.hyp_ids = [h["id"] for h in saved]

        with st.spinner("各仮説を批判エージェントが評価中..."):
            critiques = run_critique(
                st.session_state.session_id,
                st.session_state.hyp_ids,
                st.session_state.hypotheses,
                st.session_state.analyzed_text,
            )

        st.session_state.critiques = critiques
        _go("result")


# ─────────────────────────────────────────────
# 画面4: 結果画面
# ─────────────────────────────────────────────

def page_result():
    st.title("行動仮説と批判")
    st.caption("仮説を確定するか、批判をもとに再生成するかを選んでください")
    st.divider()

    hypotheses = st.session_state.hypotheses
    critiques = st.session_state.critiques

    # 仮説と批判をセットで表示する。
    # expander を使うことで批判は折りたたみにして、
    # まず仮説だけを読んでから批判を確認するという順番を促している。
    for i, (hyp, crit) in enumerate(zip(hypotheses, critiques), 1):
        st.subheader(f"仮説 {i}")
        st.write(hyp)
        with st.expander(f"批判を見る（仮説{i}）"):
            st.warning(crit)

    st.divider()

    # 再生成の上限。超えたらボタンを無効化してAPI費用の暴走を防ぐ。
    MAX_REGEN = 3
    regen_count = st.session_state.regen_count
    regen_disabled = regen_count >= MAX_REGEN

    if regen_disabled:
        st.caption(f"再生成は {MAX_REGEN} 回までです。仮説を確定してください。")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("確定する", type="primary", use_container_width=True):
            with st.spinner("保存中..."):
                # ユーザーが確定した仮説の is_final を立てる
                set_final_hypotheses(st.session_state.hyp_ids)
                complete_session(st.session_state.session_id)
            _go("done")

    with col2:
        if st.button("再生成する", use_container_width=True, disabled=regen_disabled):
            # 批判を1つの文字列に結合して生成エージェントへ渡す。
            # 各批判に仮説番号を付けることで、
            # 生成エージェントが「どの仮説のどの弱点を克服すべきか」を把握しやすくする。
            combined_criticism = "\n".join(
                f"仮説{i+1}への批判: {c}"
                for i, c in enumerate(critiques)
            )

            with st.spinner("批判をもとに仮説を再生成中..."):
                new_hypotheses = run_generation(
                    st.session_state.session_id,
                    st.session_state.analyzed_text,
                    count=3,
                    previous_criticism=combined_criticism,
                )

            if not new_hypotheses:
                st.error("再生成に失敗しました。もう一度お試しください。")
                return

            from database import get_hypotheses, get_latest_generation
            gen = get_latest_generation(st.session_state.session_id)
            saved = get_hypotheses(st.session_state.session_id, generation=gen)

            st.session_state.hypotheses = [h["content"] for h in saved]
            st.session_state.hyp_ids = [h["id"] for h in saved]

            with st.spinner("新しい仮説を評価中..."):
                new_critiques = run_critique(
                    st.session_state.session_id,
                    st.session_state.hyp_ids,
                    st.session_state.hypotheses,
                    st.session_state.analyzed_text,
                )

            st.session_state.critiques = new_critiques
            st.session_state.regen_count += 1
            st.rerun()


# ─────────────────────────────────────────────
# 画面5: 保存完了画面
# ─────────────────────────────────────────────

def page_done():
    st.title("保存完了")
    st.success("仮説を確定して記録を保存しました。")
    st.divider()

    st.subheader("確定した行動仮説")
    for i, hyp in enumerate(st.session_state.hypotheses, 1):
        st.write(f"**{i}.** {hyp}")

    st.divider()

    if st.button("新しい日記を書く", type="primary"):
        _reset()


# ─────────────────────────────────────────────
# ルーティング
# ─────────────────────────────────────────────

# page の値に応じて対応する関数を呼ぶ。
# 辞書でマッピングすることで、画面追加時にここだけ変えれば済む。
_PAGES = {
    "input": page_input,
    "deep_dive": page_deep_dive,
    "analysis": page_analysis,
    "result": page_result,
    "done": page_done,
}

page_fn = _PAGES.get(st.session_state.page, page_input)
page_fn()
