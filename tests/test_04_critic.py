"""The critic is the anti-hallucination check: every number/status/citation in an LLM draft must trace to a tool output."""
import unittest
from unittest import mock
from _env import pfe, ensure_portfolio


def pfe_result(status="BREACH", pfe_val=5_000_000.0, lim=3_500_000.0):
    return {"tool": "check_pfe_limit", "ok": True, "args": {}, "output": {
        "counterparty_id": "CP-004", "counterparty_name": "Zenith", "peak_pfe_collateralized_usd": pfe_val, "peak_pfe_uncollateralized_usd": 9_000_000.0,
        "credit_limit_usd": lim, "utilization_pct": round(pfe_val / lim * 100, 1), "status": status, "current_mtm_usd": 1_200_000.0,
        "pfe_quantile": 0.95, "horizon_days": 180}}


def margin_result(action="CALL_COUNTERPARTY", call=410_000.0, ret=0.0):
    return {"tool": "calculate_margin_call", "ok": True, "args": {}, "output": {
        "delivery_amount_usd": call, "return_amount_usd": ret, "action": action, "requires_human_approval": action != "NO_ACTION"}}


def clause_result():
    return {"tool": "csa_clause_retriever", "ok": True, "args": {}, "output": {"chunks": [{"cite": "S1", "text": "MTA is USD 250,000"}], "flags": []}}


class TestCritic(unittest.TestCase):
    def issues(self, answer, results):
        return pfe.critic(answer, results)["issues"]

    def test_good_answer_passes(self):
        a = "Zenith (CP-004) is in BREACH: peak PFE $5,000,000 vs limit $3,500,000 (142.9% utilisation)."
        self.assertEqual(pfe.critic(a, [pfe_result()]), {"ok": True, "issues": []})

    def test_invented_number_is_caught(self):
        a = "CP-004 is in BREACH: peak PFE $5,000,000 vs limit $9,999,999 (142.9%)."
        self.assertTrue(any("ungrounded numbers" in i for i in self.issues(a, [pfe_result()])))

    def test_small_rounding_differences_are_tolerated(self):
        a = "CP-004 is in BREACH: peak PFE $5,000,400 vs limit $3,500,000."          # within 0.6%
        self.assertFalse(any("ungrounded" in i for i in self.issues(a, [pfe_result()])))

    def test_missing_key_figure_or_status_is_caught(self):
        self.assertIn("missing PFE figure or status", self.issues("CP-004 has a limit of $3,500,000.", [pfe_result()]))
        self.assertIn("missing PFE figure or status", self.issues("Peak PFE is $5,000,000 for CP-004.", [pfe_result()]))   # status word absent

    def test_contradiction_within_limit_vs_breach(self):
        a = "CP-004 is within its limit; peak PFE $5,000,000 BREACH."
        self.assertTrue(any("within limit but status is BREACH" in i for i in self.issues(a, [pfe_result()])))

    def test_breach_claimed_when_tool_says_ok(self):
        r = pfe_result(status="OK", pfe_val=1_000_000.0)
        a = "CP-004 is in breach with peak PFE $1,000,000 OK."
        self.assertTrue(any("breach" in i.lower() for i in self.issues(a, [r])))

    def test_invented_status_word(self):
        r = pfe_result(status="OK", pfe_val=1_000_000.0)
        self.assertTrue(any("status 'WARNING'" in i for i in self.issues("CP-004 is on WARNING; OK peak PFE $1,000,000.", [r])))

    def test_missing_citation_on_clause_answer(self):
        self.assertIn("missing [S#] citations", self.issues("The MTA is 250,000 USD.", [clause_result()]))
        self.assertEqual(self.issues("The MTA is USD 250,000 [S1].", [clause_result()]), [])

    def test_margin_amounts(self):
        self.assertEqual(self.issues("Call the counterparty for $410,000.", [margin_result()]), [])
        self.assertIn("missing margin call / return amount", self.issues("A margin call is needed.", [margin_result()]))
        self.assertEqual(self.issues("No margin call is required.", [margin_result("NO_ACTION", 0.0)]), [])
        self.assertIn("missing margin call / return amount", self.issues("Call them for something.", [margin_result("NO_ACTION", 0.0)]))

    def test_a_failed_tool_must_be_admitted(self):
        failed = [{"tool": "check_pfe_limit", "ok": False, "error": "ToolError: Unknown counterparty"}]
        self.assertIn("a tool failed but the answer does not say so", self.issues("CP-099 is fine.", failed))
        self.assertEqual(self.issues("The check failed: unknown counterparty.", failed), [])

    def test_meta_commentary_about_missing_fields_is_flagged(self):
        a = "Peak PFE $5,000,000 BREACH; the limit is not specified in the tool output."
        self.assertTrue(any("not in the tool output" in i for i in self.issues(a, [pfe_result()])))

    def test_citation_and_date_noise_is_ignored_by_number_check(self):
        self.assertEqual(pfe._numbers("See [S2] for CP-004 on 2026-11-22"), [])
        self.assertEqual(pfe._numbers("MTA $250,000 and 12%"), [250000.0, 12.0])


class TestDeterministicFallbackAlwaysPassesCritic(unittest.TestCase):
    """If the model fails twice, the agent returns render_deterministic(); that text must itself satisfy the critic.
    Uses REAL tool outputs (not hand-made dicts) so the renderer and critic are tested against the true schema."""
    @classmethod
    def setUpClass(cls):
        ensure_portfolio()

    def _margin(self, delta):
        held = float(pfe.cp_table().query("counterparty_id == 'CP-001'").iloc[0].collateral_units)
        with mock.patch.object(pfe, "current_mtm", return_value=(held + delta, [], ["t"])):
            return {"tool": "calculate_margin_call", "ok": True, "args": {}, "output": pfe.calculate_margin_call_tool("CP-001")}

    def test_fallback_text_is_grounded(self):
        cases = {"limit BREACH": [{"tool": "check_pfe_limit", "ok": True, "args": {}, "output": pfe.check_pfe_limit_tool("CP-004")}],
                 "limit OK": [{"tool": "check_pfe_limit", "ok": True, "args": {}, "output": pfe.check_pfe_limit_tool("CP-001")}],
                 "margin call": [self._margin(900_000)], "no action": [self._margin(10_000)], "return": [self._margin(-600_000)],
                 "haircut": [{"tool": "regulatory_haircut", "ok": True, "args": {}, "output": pfe.regulatory_haircut_tool("corporate", ">5y", "A+-BBB-")}],
                 "market data": [{"tool": "get_market_data", "ok": True, "args": {}, "output": pfe.get_market_data_tool("SPY")}]}
        for name, res in cases.items():
            with self.subTest(name):
                text = pfe.render_deterministic(res)
                self.assertEqual(pfe.critic(text, res)["issues"], [], text)

    def test_failed_tool_is_reported(self):
        res = [{"tool": "check_pfe_limit", "ok": False, "error": "boom"}]
        text = pfe.render_deterministic(res)
        self.assertIn("failed", text)
        self.assertEqual(pfe.critic(text, res)["issues"], [])


if __name__ == "__main__":
    unittest.main()
