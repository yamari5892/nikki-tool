"""
nikki-tool データベースモジュール

DESIGN.md のデータフローに沿って、以下の流れで書き込みが起きる:
  1. ユーザーが日記を入力 → sessions テーブルにセッション作成
  2. 深掘り対話 → drill_messages テーブルに1往復ずつ追記
  3. 原因分析完了 → sessions.cause_analysis を更新
  4. 仮説生成 → hypotheses テーブルに追加（再生成のたびに generation を増やす）
  5. 批判生成 → criticisms テーブルに追加（各仮説に紐づく）
  6. ユーザー確定 → hypotheses.is_final を立てて sessions.status を completed に
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# DBファイルの置き場所。プロジェクトルートに固定しておくことで
# どのモジュールから呼んでも同じDBを参照できる
DB_PATH = Path(__file__).parent / "nikki.db"


def _get_conn() -> sqlite3.Connection:
    """接続を返すヘルパー。
    row_factory を設定することで cursor.fetchall() が
    辞書ライクな sqlite3.Row を返すようにする。
    エージェント側でカラム名でアクセスできて便利。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # 外部キー制約はデフォルトで無効なので明示的に有効化する
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ─────────────────────────────────────────────
# テーブル初期化
# ─────────────────────────────────────────────

def init_db() -> None:
    """テーブルが存在しなければ作成する。
    IF NOT EXISTS を使うことで、何度呼んでも副作用なし。
    アプリ起動時に毎回呼ぶ設計にしておくと初期化漏れがない。
    """
    with _get_conn() as conn:
        conn.executescript("""
            -- ① セッションテーブル
            -- 1セッション = 「日記を書いて仮説確定するまでの1サイクル」
            -- depth_level と hypothesis_count は Phase 2 で使うが、
            -- 今からスキーマに含めておくことで後から ALTER TABLE が不要になる
            CREATE TABLE IF NOT EXISTS sessions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT    NOT NULL,
                diary_text       TEXT    NOT NULL,
                depth_level      TEXT    NOT NULL DEFAULT 'normal',
                cause_analysis   TEXT,            -- 深掘り完了後に UPDATE で埋める
                hypothesis_count INTEGER NOT NULL DEFAULT 3,
                status           TEXT    NOT NULL DEFAULT 'in_progress'
                -- status: 'in_progress' | 'completed'
            );

            -- ② 深掘り対話テーブル
            -- 深掘り・分析エージェントはこのテーブルを読んで会話履歴を復元する
            -- turn_order を持つことで、取り出し時に ORDER BY turn_order すれば
            -- 時系列順が保証される（inserted order ≠ DB保存順になるケースに備える）
            CREATE TABLE IF NOT EXISTS drill_messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER NOT NULL REFERENCES sessions(id),
                role         TEXT    NOT NULL,  -- 'user' | 'assistant'
                content      TEXT    NOT NULL,
                turn_order   INTEGER NOT NULL,
                created_at   TEXT    NOT NULL
            );

            -- ③ 仮説テーブル
            -- generation カラムで「何回目の生成か」を管理する
            -- 再生成のたびに同じ session_id で generation を増やして INSERT する
            -- is_final=1 の行が最終的にユーザーが確定した仮説
            CREATE TABLE IF NOT EXISTS hypotheses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER NOT NULL REFERENCES sessions(id),
                content      TEXT    NOT NULL,
                generation   INTEGER NOT NULL DEFAULT 1,
                is_final     INTEGER NOT NULL DEFAULT 0,  -- SQLiteはBOOL非対応なので0/1で代用
                created_at   TEXT    NOT NULL
            );

            -- ④ 批判テーブル
            -- 各仮説に対して批判エージェントが出力したコメントを格納する
            -- hypothesis_id に紐づけることで、どの仮説への批判かが明確になる
            -- DESIGN.md では「各コメントは文字列」とあるため、1行=1批判コメントとする
            CREATE TABLE IF NOT EXISTS criticisms (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
                content       TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );
        """)


# ─────────────────────────────────────────────
# セッション操作
# ─────────────────────────────────────────────

def create_session(diary_text: str, depth_level: str = "normal") -> int:
    """日記入力直後に呼ぶ。セッションを作成して session_id を返す。
    深掘り・分析エージェントはこの id を受け取って以降の書き込みに使う。
    """
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (created_at, diary_text, depth_level) VALUES (?, ?, ?)",
            (now, diary_text, depth_level),
        )
        return cur.lastrowid


def update_cause_analysis(session_id: int, cause_analysis: str) -> None:
    """深掘り対話が完了したタイミングで呼ぶ。
    原因分析結果を sessions テーブルに書き込む。
    生成エージェントはこの値を読んで仮説を生成する。
    """
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET cause_analysis = ? WHERE id = ?",
            (cause_analysis, session_id),
        )


def complete_session(session_id: int) -> None:
    """ユーザーが「確定」を押したタイミングで呼ぶ。
    status を completed にすることで、検索画面で完了済みセッションだけを
    フィルタできるようにする。
    """
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET status = 'completed' WHERE id = ?",
            (session_id,),
        )


def get_session(session_id: int) -> dict | None:
    """session_id で1件取得して辞書で返す。
    見つからない場合は None を返すことで呼び出し側がエラー処理しやすい。
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def search_sessions(keyword: str) -> list[dict]:
    """過去記録の全文検索（Phase 2 で使用）。
    LIKE 検索は大量データには向かないが、個人用途であれば十分な速度。
    日記本文・原因分析の両方を対象にすることで漏れを減らす。
    """
    like = f"%{keyword}%"
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM sessions
               WHERE (diary_text LIKE ? OR cause_analysis LIKE ?)
               AND status = 'completed'
               ORDER BY created_at DESC""",
            (like, like),
        ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# 深掘り対話操作
# ─────────────────────────────────────────────

def add_drill_message(session_id: int, role: str, content: str) -> int:
    """深掘り対話の1ターン（ユーザー発言 or AI発言）を記録する。
    turn_order は既存レコード数+1にすることで、
    後から正しい順序で取り出せる。
    """
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        # 既存のターン数を数えて次の order を決定
        count = conn.execute(
            "SELECT COUNT(*) FROM drill_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        cur = conn.execute(
            """INSERT INTO drill_messages (session_id, role, content, turn_order, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, role, content, count + 1, now),
        )
        return cur.lastrowid


def get_drill_messages(session_id: int) -> list[dict]:
    """深掘り・分析エージェントに渡す会話履歴を返す。
    Anthropic API の messages 形式（role/content の辞書リスト）と
    互換性を持たせるため、role と content だけを抽出して返す。
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT role, content FROM drill_messages
               WHERE session_id = ?
               ORDER BY turn_order ASC""",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]


# ─────────────────────────────────────────────
# 仮説操作
# ─────────────────────────────────────────────

def add_hypothesis(session_id: int, content: str, generation: int = 1) -> int:
    """生成エージェントが出力した仮説を1件ずつ保存する。
    generation は再生成のたびにインクリメントして渡すことで
    「何回目の生成か」が後から追えるようになる。
    """
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO hypotheses (session_id, content, generation, created_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, content, generation, now),
        )
        return cur.lastrowid


def get_hypotheses(session_id: int, generation: int | None = None) -> list[dict]:
    """仮説リストを取得する。
    generation を指定すると特定回の生成結果だけを返す。
    省略すると全世代の仮説を返す（過去ログ表示用）。
    """
    with _get_conn() as conn:
        if generation is not None:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE session_id = ? AND generation = ?",
                (session_id, generation),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE session_id = ? ORDER BY generation, id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_latest_generation(session_id: int) -> int:
    """現在の最大 generation 番号を返す。
    再生成時に「次は何回目か」を計算するために使う。
    1件もなければ 0 を返す（呼び出し側で +1 して使う）。
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(generation) FROM hypotheses WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row[0] is not None else 0


def set_final_hypotheses(hypothesis_ids: list[int]) -> None:
    """ユーザーが確定した仮説の is_final を 1 にする。
    複数まとめて更新できるようリストで受け取る。
    """
    with _get_conn() as conn:
        conn.executemany(
            "UPDATE hypotheses SET is_final = 1 WHERE id = ?",
            [(hid,) for hid in hypothesis_ids],
        )


# ─────────────────────────────────────────────
# 批判操作
# ─────────────────────────────────────────────

def add_criticism(hypothesis_id: int, content: str) -> int:
    """批判エージェントが出力した批判コメントを1件保存する。
    DESIGN.md の「各コメントは文字列」に合わせて1行=1コメントとする。
    """
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO criticisms (hypothesis_id, content, created_at) VALUES (?, ?, ?)",
            (hypothesis_id, content, now),
        )
        return cur.lastrowid


def get_criticisms(hypothesis_id: int) -> list[dict]:
    """hypothesis_id に紐づく批判コメントを全件返す。
    結果画面で仮説と批判をセットで表示するときに使う。
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM criticisms WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_hypotheses_with_criticisms(session_id: int, generation: int) -> list[dict]:
    """結果画面用の複合取得関数。
    指定世代の仮説リストに、各仮説の批判コメントをネストして返す。
    例: [{"id": 1, "content": "...", "criticisms": ["批判1", "批判2", ...]}, ...]
    UI 層がループ1回でレンダリングできる形にまとめる。
    """
    hypotheses = get_hypotheses(session_id, generation)
    result = []
    with _get_conn() as conn:
        for h in hypotheses:
            rows = conn.execute(
                "SELECT content FROM criticisms WHERE hypothesis_id = ?",
                (h["id"],),
            ).fetchall()
            h["criticisms"] = [r["content"] for r in rows]
            result.append(h)
    return result


# ─────────────────────────────────────────────
# 動作確認用（python database.py で直接実行）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== database.py 動作確認 ===\n")

    # 既存のテスト用DBを削除してクリーンな状態で確認
    test_db = Path(__file__).parent / "nikki.db"
    if test_db.exists():
        test_db.unlink()
        print("既存のnikki.dbを削除しました\n")

    # 1. テーブル初期化
    init_db()
    print("[OK] init_db(): テーブル作成完了")

    # 2. セッション作成
    sid = create_session("今日はチームのMTGで意見を言えなかった。", depth_level="normal")
    print(f"[OK] create_session(): session_id={sid}")

    # 3. 深掘り対話の記録
    add_drill_message(sid, "assistant", "その時、なぜ意見を言えなかったと思いますか？")
    add_drill_message(sid, "user", "却下されるのが怖かったと思います。")
    add_drill_message(sid, "assistant", "過去に却下された経験がありますか？")
    add_drill_message(sid, "user", "半年前に似た提案をして一蹴されました。")
    msgs = get_drill_messages(sid)
    print(f"[OK] add_drill_message() / get_drill_messages(): {len(msgs)}件取得")
    for m in msgs:
        print(f"   [{m['role']}] {m['content'][:30]}...")

    # 4. 原因分析を更新
    update_cause_analysis(sid, "過去の失敗体験から自己防衛として発言を控える癖がついている。")
    session = get_session(sid)
    print(f"[OK] update_cause_analysis(): cause_analysis='{session['cause_analysis'][:30]}...'")

    # 5. 仮説を3件追加（generation=1）
    h_ids = []
    for text in [
        "次のMTGで小さな意見を1つだけ出す練習をする",
        "信頼できる同僚に事前に意見を話して反応を確認する",
        "却下された原因を振り返って次の提案に活かす",
    ]:
        hid = add_hypothesis(sid, text, generation=1)
        h_ids.append(hid)
        print(f"[OK] add_hypothesis(): hypothesis_id={hid}, content='{text[:20]}...'")

    # 6. 各仮説に批判を追加
    for hid in h_ids:
        add_criticism(hid, "具体的な実行タイミングが決まっていない")
        add_criticism(hid, "失敗した場合のフォローアップがない")
        add_criticism(hid, "なぜこの行動が根本原因に効くかの根拠が弱い")
    print("[OK] add_criticism(): 各仮説に3件の批判を追加")

    # 7. 複合取得で確認
    result = get_hypotheses_with_criticisms(sid, generation=1)
    print(f"\n[OK] get_hypotheses_with_criticisms(): {len(result)}件")
    for h in result:
        print(f"  仮説: {h['content'][:30]}...")
        for c in h["criticisms"]:
            print(f"    批判: {c}")

    # 8. 確定 & セッション完了
    set_final_hypotheses([h_ids[0]])
    complete_session(sid)
    session = get_session(sid)
    print(f"\n[OK] complete_session(): status='{session['status']}'")

    # 9. 検索
    found = search_sessions("MTG")
    print(f"[OK] search_sessions('MTG'): {len(found)}件ヒット")

    print("\n=== 全テスト完了 ===")
