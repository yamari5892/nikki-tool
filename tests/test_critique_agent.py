"""
tests/test_critique_agent.py

批判エージェントのテストスイート

実行方法:
  # ロジックのみ(APIコスト0)
  python -X utf8 -m pytest tests/test_critique_agent.py -v

  # API込み(実際にAPIを叩く統合テスト)
  TEST_WITH_API=1 python -X utf8 -m pytest tests/test_critique_agent.py -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import database as db_module
from agents.critique_agent import _parse_json_response, run_critique


# ─────────────────────────────────────────────
# フィクスチャ
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """テストごとにクリーンな一時 DB を使う。本番の nikki.db を汚さない。"""
    test_db_path = tmp_path / "test_nikki.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    db_module.init_db()
    yield test_db_path


@pytest.fixture
def session_with_hypotheses():
    """セッション・仮説3件をまとめて作成するフィクスチャ。
    批判エージェントのテストは必ず仮説が存在する状態から始まるため、
    共通のセットアップをここにまとめた。
    """
    sid = db_module.create_session("MTGで意見を言えなかった", depth_level="normal")
    db_module.update_cause_analysis(sid, "過去の失敗体験から回避パターンが形成されている")

    hypotheses = ["仮説A: 行動層", "仮説B: 思考層", "仮説C: 環境層"]
    hyp_ids = [db_module.add_hypothesis(sid, h, generation=1) for h in hypotheses]

    return {"session_id": sid, "hyp_ids": hyp_ids, "hypotheses": hypotheses}


ANALYZED_TEXT = "【日記】MTGで発言できなかった。\n【原因分析】否定されるという信念がある。"


# ─────────────────────────────────────────────
# 単体テスト: _parse_json_response
# ─────────────────────────────────────────────

class TestParseJsonResponse:

    def test_正常なJSONを解析できる(self):
        raw = '{"critiques": ["批判A", "批判B", "批判C"]}'
        result = _parse_json_response(raw)
        assert result["critiques"] == ["批判A", "批判B", "批判C"]

    def test_コードブロック包みのJSONを解析できる(self):
        raw = '```json\n{"critiques": ["批判A"]}\n```'
        result = _parse_json_response(raw)
        assert result["critiques"] == ["批判A"]

    def test_前後に余分なテキストがあるJSONを解析できる(self):
        raw = '以下が批判です:\n{"critiques": ["批判A"]}\n以上。'
        result = _parse_json_response(raw)
        assert result["critiques"] == ["批判A"]

    def test_完全に壊れたレスポンスは空リストを返す(self):
        raw = "これはJSONではありません"
        result = _parse_json_response(raw)
        assert result == {"critiques": []}


# ─────────────────────────────────────────────
# 単体テスト: run_critique (APIをモック)
# ─────────────────────────────────────────────

class TestRunCritiqueMocked:

    def _make_mock_response(self, critiques: list[str]):
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps({"critiques": critiques})
        return mock_response

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_批判3件を返す(self, mock_cls, session_with_hypotheses):
        expected = ["批判A", "批判B", "批判C"]
        mock_cls.return_value.messages.create.return_value = (
            self._make_mock_response(expected)
        )
        d = session_with_hypotheses
        result = run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)
        assert result == expected

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_各批判がhypothesis_idに紐づいてDBに保存される(self, mock_cls, session_with_hypotheses):
        mock_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["批判A", "批判B", "批判C"])
        )
        d = session_with_hypotheses
        run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)

        for hid, expected_text in zip(d["hyp_ids"], ["批判A", "批判B", "批判C"]):
            saved = db_module.get_criticisms(hid)
            assert len(saved) == 1
            assert saved[0]["content"] == expected_text

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_件数不一致のときDBに保存しない(self, mock_cls, session_with_hypotheses):
        # 仮説3件に対して批判が2件しか返ってきた場合、対応がずれてDBが壊れるため保存しない
        mock_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["批判A", "批判B"])  # 2件しかない
        )
        d = session_with_hypotheses
        run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)

        for hid in d["hyp_ids"]:
            assert db_module.get_criticisms(hid) == []

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_仮説が空のときAPIを呼ばずに空リストを返す(self, mock_cls, session_with_hypotheses):
        d = session_with_hypotheses
        result = run_critique(d["session_id"], [], [], ANALYZED_TEXT)

        assert result == []
        # 仮説が空なので API は呼ばれないはず
        mock_cls.return_value.messages.create.assert_not_called()

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_APIが空レスポンスを返しても空リストで返る(self, mock_cls, session_with_hypotheses):
        mock_response = MagicMock()
        mock_response.content[0].text = "これはJSONではありません"
        mock_cls.return_value.messages.create.return_value = mock_response

        d = session_with_hypotheses
        result = run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)

        assert result == []

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_ユーザーメッセージに仮説と文脈が含まれる(self, mock_cls, session_with_hypotheses):
        """APIに渡すユーザーメッセージに仮説テキストと分析テキストが両方含まれているかを確認。
        批判エージェントが文脈なしに仮説だけを見ていないことを保証する。
        """
        mock_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["批判A", "批判B", "批判C"])
        )
        d = session_with_hypotheses
        run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)

        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]

        # 文脈(原因分析)が含まれていること
        assert "原因分析" in user_content
        # 全仮説が含まれていること
        for h in d["hypotheses"]:
            assert h in user_content

    @patch("agents.critique_agent.anthropic.Anthropic")
    def test_再実行で追加の批判が同じhypothesisに蓄積される(self, mock_cls, session_with_hypotheses):
        """同じ hypothesis_id に対して run_critique を2回実行すると批判が2件になることを確認。
        再生成→再批判のループで批判が上書きではなく追記されることを保証する。
        """
        mock_cls.return_value.messages.create.return_value = (
            self._make_mock_response(["批判A", "批判B", "批判C"])
        )
        d = session_with_hypotheses
        run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)
        run_critique(d["session_id"], d["hyp_ids"], d["hypotheses"], ANALYZED_TEXT)

        # 同じ hypothesis に2回保存されるので2件になる
        saved = db_module.get_criticisms(d["hyp_ids"][0])
        assert len(saved) == 2


# ─────────────────────────────────────────────
# 統合テスト: 実際のAPIを使う(TEST_WITH_API=1 のときのみ)
# ─────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("TEST_WITH_API"),
    reason="TEST_WITH_API=1 を設定したときだけ実行(APIコスト節約)",
)
class TestRunCritiqueWithRealAPI:

    HYPOTHESES = [
        "[行動層] 次回MTGの最初の5分以内に一言だけ感想を述べる",
        "[思考層] 提案=否定されるという信念を議論の材料を提供する行為と再定義する",
        "[環境層] MTG前日にSlackで意見メモを共有するルーティンを作る",
    ]

    def test_APIで仮説と同数の批判が返る(self, session_with_hypotheses):
        d = session_with_hypotheses
        result = run_critique(d["session_id"], d["hyp_ids"], self.HYPOTHESES, ANALYZED_TEXT)
        assert len(result) == len(self.HYPOTHESES)

    def test_APIで批判がDBに保存される(self, session_with_hypotheses):
        d = session_with_hypotheses
        run_critique(d["session_id"], d["hyp_ids"], self.HYPOTHESES, ANALYZED_TEXT)
        for hid in d["hyp_ids"]:
            saved = db_module.get_criticisms(hid)
            assert len(saved) == 1
            assert len(saved[0]["content"]) > 0
