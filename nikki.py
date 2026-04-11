import anthropic
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

def load_records():
    """過去の記録を読み込む"""
    if os.path.exists("records.json"):
        with open("records.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_records(records):
    """記録を保存する"""
    with open("records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def show_records(records):
    """過去の記録を一覧表示する"""
    if not records:
        print("まだ記録がありません。")
        return
    print("\n=== 過去の記録 ===\n")
    for r in records:
        print(f"📅 {r['date']}")
        print(f"  やったこと : {r['did']}")
        print(f"  学んだこと : {r['learned']}")
        print(f"  気持ち     : {r['feeling']}")
        print(f"  明日の予定 : {r['tomorrow']}")
        if "feedback" in r:
            print(f"  AIの一言   : {r['feedback'][:50]}...")
        print()

def get_feedback(entry, past_records):
    """Claude APIからフィードバックをもらう"""
    client = anthropic.Anthropic(api_key=API_KEY)

    past_summary = ""
    if past_records:
        past_summary = "【過去の記録】\n"
        for r in past_records[-5:]:
            past_summary += f"- {r['date']}: {r['did']} / 学び: {r['learned']}\n"

    prompt = f"""あなたは個人の成長をサポートするコーチです。
以下の今日の記録と過去の記録をもとに、成長の気づきや明日へのアドバイスを200字程度で日本語でフィードバックしてください。
厳しくも温かく、具体的にお願いします。

{past_summary}

【今日の記録】
やったこと: {entry['did']}
学んだこと: {entry['learned']}
気持ち・感情: {entry['feeling']}
明日やること: {entry['tomorrow']}
"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

def main():
    print("=== 個人成長記録ツール ===\n")
    print("1. 今日の記録をつける")
    print("2. 過去の記録を振り返る")
    choice = input("\n番号を選んでください: ")

    records = load_records()

    if choice == "2":
        show_records(records)
        return

    # 記録入力
    print("\n今日の記録をつけましょう。\n")
    did = input("今日やったこと: ")
    learned = input("今日学んだこと: ")
    feeling = input("今日の気持ち・感情: ")
    tomorrow = input("明日やること: ")

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "did": did,
        "learned": learned,
        "feeling": feeling,
        "tomorrow": tomorrow
    }

    print("\n--- Claude からのフィードバック ---")
    feedback = get_feedback(entry, records)
    print(feedback)

    entry["feedback"] = feedback
    records.append(entry)
    save_records(records)

    print("\n記録を保存しました。")

if __name__ == "__main__":
    main()