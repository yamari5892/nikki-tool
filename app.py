import streamlit as st
import anthropic
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

def load_records():
    if os.path.exists("records.json"):
        with open("records.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_records(records):
    with open("records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def chat_with_coach(messages):
    """コーチとして対話し、答えの深さを判断して次の問いを返す"""
    client = anthropic.Anthropic(api_key=API_KEY)

    system_prompt = """あなたは個人の成長を最大化するコーチです。
ユーザーの今日の経験から、深い気づきを引き出すことがあなたの役割です。

以下のルールに従って対話してください：

1. ユーザーの答えが浅い・曖昧な場合は、次の質問に進まず深掘りしてください
   例：「もう少し具体的に教えてください」「その時どんな状況でしたか？」

2. 答えに十分な具体性・感情・気づきが含まれている場合のみ次のテーマへ進んでください

3. 以下の流れで対話を進めてください：
   ① 今日どんな経験をしたか（出来事）
   ② その時何を感じたか（感情）
   ③ そこから何を学んだか（気づき）
   ④ 明日どう活かすか（行動）
   
4. 全てのテーマが深く掘り下げられたら、最後に成長の気づきを200字程度でまとめてください
   まとめの前に必ず「【本日の成長まとめ】」という見出しをつけてください

5. 常に日本語で、温かくも真剣に関わってください"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )
    return response.content[0].text

# ページ設定
st.set_page_config(page_title="個人成長記録ツール", page_icon="🌱")
st.title("🌱 個人成長記録ツール")
st.caption("今日の経験から、深い気づきを引き出そう")

# セッション初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "saved" not in st.session_state:
    st.session_state.saved = False
if "started" not in st.session_state:
    st.session_state.started = False

# 開始前の画面
if not st.session_state.started:
    st.markdown("### 今日もお疲れ様でした。")
    st.markdown("コーチとの対話を通じて、今日の経験を深く振り返りましょう。")
    if st.button("振り返りをはじめる", type="primary"):
        st.session_state.started = True
        opening = "こんにちは！今日一日お疲れ様でした。\n\nまず、今日どんな経験や出来事がありましたか？印象に残っていることを自由に話してください。"
        st.session_state.messages.append({
            "role": "assistant",
            "content": opening
        })
        st.rerun()

# チャット画面
else:
    # 過去のメッセージを表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # まとめが出たら保存ボタンを表示
    last_msg = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
    if "【本日の成長まとめ】" in last_msg and not st.session_state.saved:
        if st.button("今日の記録を保存する", type="primary"):
            records = load_records()
            records.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "conversation": st.session_state.messages,
                "summary": last_msg
            })
            save_records(records)
            st.session_state.saved = True
            st.success("記録を保存しました！")

    # 保存済みなら新しいセッションボタン
    if st.session_state.saved:
        if st.button("新しい振り返りをはじめる"):
            st.session_state.messages = []
            st.session_state.saved = False
            st.session_state.started = False
            st.rerun()

    # 入力欄
    if not st.session_state.saved:
        user_input = st.chat_input("今日のことを話してください...")
        if user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("考えています..."):
                    response = chat_with_coach(st.session_state.messages)
                st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()