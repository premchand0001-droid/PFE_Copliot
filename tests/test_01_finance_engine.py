"""Deterministic finance: haircuts, margin-call arithmetic, PFE simulation, stress. The LLM never does this arithmetic,
so these tests are exact (hand-computed expected values)."""
import unittest
from unittest import mock
import numpy as np, pandas as pd
from _env import pfe, ensure_portfolio


class TestHaircuts(unittest.TestCase):
    def test_known_values(self):
        cases = [(("cash",), 0.0), (("government", "1-5y", "AAA-AA-"), 2.0), (("government", ">5y", "A+-BBB-"), 6.0),
                 (("corporate", "<=1y", "AAA-AA-"), 1.0), (("corporate", ">5y", "A+-BBB-"), 12.0),
                 (("equity_main_index",), 15.0), (("equity_other",), 25.0), (("gold",), 15.0)]
        for args, exp in cases:
            with self.subTest(args=args):
                self.assertEqual(pfe.haircut_pct(*args), exp)

    def test_longer_maturity_and_lower_rating_never_cheaper(self):
        for ac in ("government", "corporate"):
            for rating in pfe.RATINGS:
                v = [pfe.haircut_pct(ac, m, rating) for m in pfe.MATURITIES]
                self.assertEqual(v, sorted(v), f"{ac}/{rating} haircut must not fall with maturity")
            for m in pfe.MATURITIES:
                self.assertGreaterEqual(pfe.haircut_pct(ac, m, "A+-BBB-"), pfe.haircut_pct(ac, m, "AAA-AA-"))

    def test_unknown_buckets_fall_back_safely(self):
        self.assertEqual(pfe.haircut_pct("government", "weird", "weird"), pfe.haircut_pct("government", "1-5y", "AAA-AA-"))

    def test_invalid_asset_class_is_a_tool_error(self):
        with self.assertRaises(pfe.ToolError):
            pfe.haircut_pct("bitcoin")

    def test_tool_output_shape(self):
        o = pfe.regulatory_haircut_tool("corporate", ">5y", "A+-BBB-")
        self.assertEqual(o["haircut_pct"], 12.0)
        self.assertEqual(o["fx_mismatch_addon_pct"], 8.0)


class TestRoundingAndStatus(unittest.TestCase):
    def test_round_up_down(self):
        self.assertEqual(pfe._round_up(1_230_001, 10_000), 1_240_000)
        self.assertEqual(pfe._round_up(1_230_000, 10_000), 1_230_000)     # exact multiple is unchanged
        self.assertEqual(pfe._round_down(1_239_999, 10_000), 1_230_000)
        self.assertEqual(pfe._round_down(1_230_000, 10_000), 1_230_000)
        self.assertEqual(pfe._round_up(123.4, 0), 123.4)                  # no rounding multiple

    def test_status_boundaries(self):
        for util, exp in [(0, "OK"), (84.99, "OK"), (85, "WARNING"), (100, "WARNING"), (100.01, "BREACH"), (250, "BREACH")]:
            with self.subTest(util=util):
                self.assertEqual(pfe._status(util), exp)


class TestMarginCall(unittest.TestCase):
    """CSA mechanics: CSA amount = max(0, MtM + IA - Threshold); Delivery = CSA - collateral held; MTA test; round up/down."""
    @classmethod
    def setUpClass(cls):
        ensure_portfolio()
        cls.cp1 = pfe.cp_table().query("counterparty_id == 'CP-001'").iloc[0]      # cash collateral, 0 haircut, TH 0, IA 0, MTA 250k, rounding 10k

    def _call(self, cp_id, mtm):
        with mock.patch.object(pfe, "current_mtm", return_value=(mtm, [], ["test"])):
            return pfe.calculate_margin_call_tool(cp_id)

    def test_call_rounded_up(self):
        held = float(self.cp1.collateral_units)                      # cash: price 1, haircut 0
        o = self._call("CP-001", held + 400_001)
        self.assertEqual(o["action"], "CALL_COUNTERPARTY")
        self.assertEqual(o["delivery_amount_usd"], 410_000.0)        # 400,001 rounded UP to the 10k multiple
        self.assertEqual(o["return_amount_usd"], 0.0)
        self.assertTrue(o["requires_human_approval"])

    def test_below_mta_means_no_action(self):
        held = float(self.cp1.collateral_units)
        o = self._call("CP-001", held + 100_000)                    # 100k < MTA 250k
        self.assertEqual(o["action"], "NO_ACTION")
        self.assertEqual(o["delivery_amount_usd"], 0.0)
        self.assertFalse(o["requires_human_approval"])

    def test_exactly_at_mta_triggers(self):
        held = float(self.cp1.collateral_units)
        o = self._call("CP-001", held + 250_000)
        self.assertEqual(o["delivery_amount_usd"], 250_000.0)

    def test_return_rounded_down(self):
        held = float(self.cp1.collateral_units)
        o = self._call("CP-001", held - 300_009)
        self.assertEqual(o["action"], "RETURN_COLLATERAL")
        self.assertEqual(o["return_amount_usd"], 300_000.0)          # rounded DOWN
        self.assertEqual(o["delivery_amount_usd"], 0.0)

    def test_counterparty_may_call_us_when_we_owe(self):
        o = self._call("CP-001", -600_001)                          # we are out-of-the-money by 600,001, own threshold 0
        self.assertEqual(o["counterparty_may_call_us_usd"], 610_000.0)

    def test_negative_mtm_gives_zero_csa_amount(self):
        o = self._call("CP-001", -5_000_000)
        self.assertEqual(o["credit_support_amount_usd"], 0.0)

    def test_invariants_for_every_counterparty_and_many_mtms(self):
        for cp_id in pfe.cp_table().counterparty_id:
            cp = pfe.cp_table().query("counterparty_id == @cp_id").iloc[0]
            for mtm in (-9e6, -1e6, 0, 3e5, 1e6, 2.5e6, 8e6, 3e7):
                with self.subTest(cp=cp_id, mtm=mtm):
                    o = self._call(cp_id, mtm)
                    self.assertGreaterEqual(o["delivery_amount_usd"], 0)
                    self.assertGreaterEqual(o["return_amount_usd"], 0)
                    self.assertFalse(o["delivery_amount_usd"] and o["return_amount_usd"], "never both a call and a return")
                    self.assertAlmostEqual(o["credit_support_amount_usd"],
                                           max(0.0, mtm + cp.independent_amount_usd - cp.threshold_usd), places=1)
                    for k in ("delivery_amount_usd", "return_amount_usd"):
                        if o[k]:
                            self.assertGreaterEqual(o[k], cp.mta_usd)
                            self.assertAlmostEqual(o[k] % cp.rounding_usd, 0, places=4)
                    self.assertEqual(o["requires_human_approval"], o["action"] != "NO_ACTION")

    def test_unknown_counterparty(self):
        with self.assertRaises(pfe.ToolError):
            pfe.calculate_margin_call_tool("CP-999")


class TestCollateralModel(unittest.TestCase):
    def _cp(self, th, mta, ia, mpor=5):
        return pd.Series(dict(counterparty_id="T", mpor_days=mpor, threshold_usd=th, mta_usd=mta, independent_amount_usd=ia))

    def _V(self, seed=0, T=30, n=2000):
        rng = np.random.default_rng(seed)
        V = np.cumsum(rng.normal(0, 1e6, (T + 1, n)), axis=0)
        V[0] = 2e6
        return V

    def test_collateralised_never_exceeds_uncollateralised(self):
        for th, mta, ia in [(0, 0, 0), (1e6, 1e5, 0), (0, 0, 5e6), (5e6, 5e5, 1e6)]:
            with self.subTest(th=th):
                r = pfe._collateralise(self._V(), self._cp(th, mta, ia), None, 2000)
                self.assertLessEqual(r["peak_pfe_collateralized_usd"], r["peak_pfe_uncollateralized_usd"] + 1e-6)

    def test_huge_threshold_means_no_collateral_benefit(self):
        r = pfe._collateralise(self._V(), self._cp(1e12, 0, 0), None, 2000)
        self.assertAlmostEqual(r["peak_pfe_collateralized_usd"], r["peak_pfe_uncollateralized_usd"], places=4)

    def test_big_independent_amount_wipes_out_exposure(self):
        r = pfe._collateralise(self._V(), self._cp(0, 0, 1e12), None, 2000)
        self.assertEqual(r["peak_pfe_collateralized_usd"], 0.0)

    def test_lower_threshold_never_increases_pfe(self):
        hi = pfe._collateralise(self._V(), self._cp(3e6, 0, 0), None, 2000)["peak_pfe_collateralized_usd"]
        lo = pfe._collateralise(self._V(), self._cp(0, 0, 0), None, 2000)["peak_pfe_collateralized_usd"]
        self.assertLessEqual(lo, hi + 1e-6)

    def test_longer_mpor_never_reduces_pfe(self):
        short = pfe._collateralise(self._V(), self._cp(0, 0, 0, 1), 1, 2000)["peak_pfe_collateralized_usd"]
        long_ = pfe._collateralise(self._V(), self._cp(0, 0, 0, 20), 20, 2000)["peak_pfe_collateralized_usd"]
        self.assertGreaterEqual(long_, short - 1e-6)


class TestSimulationAndLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_portfolio()

    def test_same_seed_is_reproducible(self):
        pfe._CACHE.clear()
        a = pfe.simulate_exposure("CP-001", seed=5)["peak_pfe_collateralized_usd"]
        pfe._CACHE.clear()
        b = pfe.simulate_exposure("CP-001", seed=5)["peak_pfe_collateralized_usd"]
        self.assertEqual(a, b)

    def test_higher_vol_gives_higher_uncollateralised_pfe(self):
        base = pfe.simulate_exposure("CP-001")["peak_pfe_uncollateralized_usd"]
        stressed = pfe.simulate_exposure("CP-001", vol_mult=2.0)["peak_pfe_uncollateralized_usd"]
        self.assertGreater(stressed, base)

    def test_pfe_bounds_and_quantile(self):
        for cp_id in pfe.cp_table().counterparty_id:
            r = pfe.simulate_exposure(cp_id)
            with self.subTest(cp=cp_id):
                self.assertGreaterEqual(r["peak_pfe_collateralized_usd"], 0)
                self.assertLessEqual(r["peak_pfe_collateralized_usd"], r["peak_pfe_uncollateralized_usd"] + 1e-6)
                self.assertEqual(r["quantile"], 0.95)

    def test_limit_check_output_is_consistent(self):
        for cp_id in pfe.cp_table().counterparty_id:
            o = pfe.check_pfe_limit_tool(cp_id)
            with self.subTest(cp=cp_id):
                self.assertAlmostEqual(o["utilization_pct"], o["peak_pfe_collateralized_usd"] / o["credit_limit_usd"] * 100, delta=0.06)
                self.assertEqual(o["status"], pfe._status(o["peak_pfe_collateralized_usd"] / o["credit_limit_usd"] * 100))

    def test_limit_set_below_pfe_is_a_breach(self):
        # CP-004 limit was built as 0.70 x modelled PFE, so it must be in BREACH
        self.assertEqual(pfe.check_pfe_limit_tool("CP-004")["status"], "BREACH")
        # CP-001 limit was built as 1.50 x modelled PFE, so it must be OK
        self.assertEqual(pfe.check_pfe_limit_tool("CP-001")["status"], "OK")

    def test_name_and_id_resolve_to_same_counterparty(self):
        self.assertEqual(pfe.resolve_counterparty("CP-3").counterparty_id, "CP-003")
        self.assertEqual(pfe.resolve_counterparty("cp 003").counterparty_id, "CP-003")
        self.assertEqual(pfe.resolve_counterparty("Meridian").counterparty_id, "CP-002")
        self.assertEqual(pfe.resolve_counterparty("zenith hedge fund llc").counterparty_id, "CP-004")
        with self.assertRaises(pfe.ToolError):
            pfe.resolve_counterparty("Lehman Brothers")

    def test_market_data_tool(self):
        o = pfe.get_market_data_tool(" spy ")
        self.assertEqual(o["ticker"], "SPY")
        self.assertGreater(o["spot"], 0)
        self.assertGreater(o["ann_vol_1y_pct"], 0)


class TestStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_portfolio()

    def _ret(self, **shock):
        r = {t: 0.0 for t in pfe.portfolio_tickers()}
        r.update(shock)
        return r

    def test_equity_crash_hurts_long_equity_and_helps_short_equity(self):
        cps = pfe.cp_table().set_index("counterparty_id")
        r1 = pfe.stress_counterparty(cps.loc["CP-001"].rename("x").to_frame().T.assign(counterparty_id="CP-001").iloc[0], self._ret(SPY=-0.4))
        self.assertLess(r1["mtm_change_usd"], 0)             # CP-001 is long SPY
        r4 = pfe.stress_counterparty(cps.loc["CP-004"].rename("x").to_frame().T.assign(counterparty_id="CP-004").iloc[0], self._ret(SPY=-0.4))
        self.assertGreater(r4["mtm_change_usd"], 0)          # CP-004 is short SPY (and EEM is left unshocked here)

    def test_zero_shock_changes_nothing(self):
        cp = pfe.cp_table().iloc[0]
        r = pfe.stress_counterparty(cp, self._ret())
        self.assertAlmostEqual(r["mtm_change_usd"], 0, places=4)
        self.assertAlmostEqual(r["collateral_change_usd"], 0, places=4)

    def test_unknown_scenario_rejected(self):
        with self.assertRaises(pfe.ToolError):
            pfe.scenario_returns("zombie_apocalypse", pfe.portfolio_tickers())

    def test_every_named_scenario_returns_a_return_for_each_ticker(self):
        tk = pfe.portfolio_tickers()
        for name in pfe.SCENARIO_NAMES:
            with self.subTest(scenario=name):
                ret, desc, notes = pfe.scenario_returns(name, tk)
                self.assertTrue(set(tk) <= set(ret))
                self.assertTrue(desc)

    def test_scenario_name_from_free_text(self):
        for q, exp in [("what if covid happened again", "covid_2020"), ("replay the 2008 financial crisis", "gfc_2008"),
                       ("2022 rate hikes", "rates_2022"), ("severe crash", "equity_crash_40"), ("some crash", "equity_crash_20")]:
            with self.subTest(q=q):
                self.assertEqual(pfe.scenario_from_text(q), exp)

    def test_stress_tool_output(self):
        o = pfe.run_stress_test_tool("equity_crash_40", write_file=False)
        self.assertEqual(o["scenario"], "equity_crash_40")
        self.assertGreaterEqual(o["liquidity_need_usd"], 0)
        self.assertGreaterEqual(o["n_breach_stressed"], 0)


if __name__ == "__main__":
    unittest.main()
