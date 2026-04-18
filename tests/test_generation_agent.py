"""
tests/test_generation_agent.py

生成エージェントのテストスイート

テスト方針:
  - API を実際に叩くテスト(統合テスト)と、ロジックだけを確認するテスト(単体テスト)を分ける
  - API テストは TEST_WITH_API=1 の環境変数があるときだけ実行する
    (CI や普段の単体確認でコストをかけないため)
  - DB テストはテスト用の一時 DB を使い、テスト後に削除する

実行方法:
  # ロジックのみ(APIコスト0)
  python -X utf8 -m pytest tests/test_generation_agent.py -v

  # API込み(実際にAPIを叩く統合テスト)
  TEST_WITH_API=1 python -X utf8 -m pytest tests/test_generation_agent.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートを sys.path に追加して database / agents を import できるようにする
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import database as db_module
from agents.generation_agent import _parse_json_response, run_generation


# ─────────────────────────────────────────────
# フィクスチャ: テスト専用の一時 DB
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """テストごとにクリーンな一時 DB を使う。
    monkeypatch で DB_PATH を書き換えることで、
    本番の nikki.db を汚さずに済む。
    """
    test_db_path = tmp_path / "test_nikki.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    db_module.init_db()
    yield test_db_path
    # tmp_path は pytest が自動クリーンアップするので明示的な削除不要


@pytest.fixture
def session_id():
    """テスト用セッションを1件作成して ID を返すフィクスチャ。"""
    sid = db_module.create_session(
        diary_text="今日のMTGで発言できなかった。",
        depth_level="normal",
    )
    db_module.update_cause_analysis(
        sid,
        "過去の失敗体験から「発言=否定される」という信念が形成されており、自己防衛として沈黙を選んでいる。",
    )
    return sid


# ─────────────────────────────────────────────
# 単体テスト: _parse_json_response
# ─────────────────────────────────────────────

class TestParseJsonResponse:
    """JSON パース処理の単体テスト。
    API レスポンスのフォーマットが揺れても正しく動くかを確認する。
    """

    def test_正常なJSONを解析できる(self):
        raw = '{"hypotheses": ["仮説A", "仮説B", "仮説C"]}'
        result = _parse_json_response(raw)
        assert result["hypotheses"] == ["仮説A", "仮説B", "仮説C"]

    def test_コードブロックに包まれたJSONを解析できる(self):
        # Claude が稀に ```json ... ``` で返すケースに対応しているかを確認
        raw = '```json\n{"hypotheses": ["仮説A", "仮説B"]}\n```'
        result = _parse_json_response(raw)
        assert result["hypotheses"] == ["仮説A", "仮説B"]

    def test_前後に余分なテキストがあるJSONを解析できる(self):
        raw = 'こちらが仮説です:\n{"hypotheses": ["仮説A"]}\n以上です。'
        result = _parse_json_response(raw)
        assert result["hypotheses"] == ["仮説A"]

    def test_完全に壊れたレスポンスは空リストを返す(self):
        # 最終フォールバック: 解析不能な場合でもクラッシュしないことを確認
        raw = "これはJSONではありません"
        result = _parse_json_response(raw)
        assert result == {"hypotheses": []}


# ─────────────────────────────────────────────
# 単体テスト: run_generation (APIをモック)
# ─────────────────────────────────────────────

class TestRunGenerationMocked:
    """API 呼び出しをモックして run_generation のロジックをテストする。
    API コストをかけずに「DBへの保存」「generation 管理」「返り値」を検証できる。
    """

    def _make_mock_response(self, hypotheses: list[str]):
        """Anthropic API レスポンスのモックオブジェクトを作る。"""
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps({"hypotheses": hypotheses})
        return mock_response

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_初回生成で仮説3件を返す(self, mock_anthropic_cls, session_id):
        expected = ["仮説1テキスト", "仮説2テキスト", "仮説3テキスト"]
        mock_anthropic_cls.return_value.messages.create.return_value = (
            self._make_mock_response(expected)
        )

        result = run_generation(session_id, "テスト用の分析テキスト", count=3)

        assert result == expected

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_初回生成でgeneration1として保存される(self, mock_anthropic_cls, session_id):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["仮説A", "仮説B", "仮説C"])
        )

        run_generation(session_id, "テスト用の分析テキスト")

        saved = db_module.get_hypotheses(session_id, generation=1)
        assert len(saved) == 3
        assert saved[0]["content"] == "仮説A"

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_再生成でgeneration2として保存される(self, mock_anthropic_cls, session_id):
        # 1回目
        mock_anthropic_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["旧仮説A", "旧仮説B", "旧仮説C"])
        )
        run_generation(session_id, "テスト用の分析テキスト")

        # 2回目(再生成)
        mock_anthropic_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["新仮説A", "新仮説B", "新仮説C"])
        )
        run_generation(session_id, "テスト用の分析テキスト", previous_criticism="批判テキスト")

        gen1 = db_module.get_hypotheses(session_id, generation=1)
        gen2 = db_module.get_hypotheses(session_id, generation=2)
        assert len(gen1) == 3
        assert len(gen2) == 3
        assert gen2[0]["content"] == "新仮説A"

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_再生成時は批判がシステムプロンプトに含まれる(self, mock_anthropic_cls, session_id):
        """再生成時に previous_criticism がシステムプロンプトに渡されているかを確認。
        モックの call_args からシステムプロンプトの内容を検証する。
        """
        mock_anthropic_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["新仮説A", "新仮説B", "新仮説C"])
        )

        criticism = "仮説が抽象的すぎて実行できない"
        run_generation(session_id, "テスト分析", previous_criticism=criticism)

        # messages.create に渡された system 引数を取り出す
        call_kwargs = mock_anthropic_cls.return_value.messages.create.call_args.kwargs
        assert criticism in call_kwargs["system"]

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_批判なしの初回はシステムプロンプトに再生成指示が含まれない(
        self, mock_anthropic_cls, session_id
    ):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["仮説A", "仮説B", "仮説C"])
        )

        run_generation(session_id, "テスト分析", previous_criticism="")

        call_kwargs = mock_anthropic_cls.return_value.messages.create.call_args.kwargs
        # 再生成指示ブロックのキーワードが含まれていないことを確認
        assert "前回の批判" not in call_kwargs["system"]

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_APIが空レスポンスを返しても空リストで返る(self, mock_anthropic_cls, session_id):
        """JSONパース失敗時にクラッシュせず空リストを返すフォールバックの確認。"""
        mock_response = MagicMock()
        mock_response.content[0].text = "これはJSONではありません"
        mock_anthropic_cls.return_value.messages.create.return_value = mock_response

        result = run_generation(session_id, "テスト分析")

        assert result == []

    @patch("agents.generation_agent.anthropic.Anthropic")
    def test_latest_generationがcountを反映して増加する(self, mock_anthropic_cls, session_id):
        """generate を3回実行するとlatest_generationが3になることを確認。"""
        for i in range(3):
            mock_anthropic_cls.return_value.messages.create.return_value = (
                self._make_mock_response([f"仮説{i}-1", f"仮説{i}-2", f"仮説{i}-3"])
            )
            run_generation(session_id, "テスト分析")

        assert db_module.get_latest_generation(session_id) == 3


# ─────────────────────────────────────────────
# 統合テスト: 実際のAPIを使う(TEST_WITH_API=1 のときのみ)
# ─────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("TEST_WITH_API"),
    reason="TEST_WITH_API=1 を設定したときだけ実行(APIコスト節約)",
)
class TestRunGenerationWithRealAPI:
    """実際の Anthropic API を使った統合テスト。
    通常の pytest 実行ではスキップされる。
    """

    ANALYZED_TEXT = """\
【日記本文(深掘り済み)】
今日のMTGで意見を言えなかった。発言しようとしたが「どうせ否定される」と感じて黙った。
去年の提案が一言で否定された経験が今も影響している。

【原因分析】
「提案=否定される」という信念が形成されており、自己防衛として沈黙を選ぶパターンがある。
"""

    def test_APIで3件の仮説が生成される(self, session_id):
        result = run_generation(session_id, self.ANALYZED_TEXT, count=3)
        assert len(result) == 3
        for h in result:
            assert isinstance(h, str)
            assert len(h) > 0

    def test_APIでDBに正しく保存される(self, session_id):
        run_generation(session_id, self.ANALYZED_TEXT, count=3)
        saved = db_module.get_hypotheses(session_id, generation=1)
        assert len(saved) == 3

    def test_APIで再生成が正しく動く(self, session_id):
        run_generation(session_id, self.ANALYZED_TEXT, count=3)
        criticism = "仮説が抽象的すぎて実行タイミングが不明"
        result2 = run_generation(
            session_id, self.ANALYZED_TEXT, count=3, previous_criticism=criticism
        )
        assert len(result2) == 3
        assert db_module.get_latest_generation(session_id) == 2
