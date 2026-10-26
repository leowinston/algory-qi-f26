"""Homework 5 self-check.   python check_hw05.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 5 self-check", "hw05.py")
m = c.load(__file__)


K, PREM = 100.0, 6.0

c.check("long call, far below the strike",
        lambda: m.payoff("long_call", K, PREM, 80), -6.0,
        note="worthless at expiry, so you lose exactly the premium")
c.check("long call, at the strike",
        lambda: m.payoff("long_call", K, PREM, 100), -6.0,
        note="intrinsic is zero at the strike, so still just the premium")
c.check("long call, well above the strike",
        lambda: m.payoff("long_call", K, PREM, 120), 14.0,
        note="(120 - 100) - 6")
c.check("long put, far below the strike",
        lambda: m.payoff("long_put", K, PREM, 80), 14.0,
        note="(100 - 80) - 6")
c.check("short call, well above the strike",
        lambda: m.payoff("short_call", K, PREM, 120), -14.0,
        note="the exact mirror of the long call")
c.check("short put, far below the strike",
        lambda: m.payoff("short_put", K, PREM, 80), -14.0,
        note="the exact mirror of the long put")
c.check("short put, above the strike",
        lambda: m.payoff("short_put", K, PREM, 120), 6.0,
        note="expires worthless, you keep the premium")

c.assert_true("long and short are exact mirrors at every price",
              lambda: all(abs(m.payoff("long_call", K, PREM, s)
                              + m.payoff("short_call", K, PREM, s)) < 1e-9
                          for s in range(60, 141, 5)),
              note="a long and a short of the same contract must sum to zero everywhere")
c.assert_true("a long put cannot gain more than the strike less the premium",
              lambda: max(m.payoff("long_put", K, PREM, s) for s in range(0, 141)) <= K - PREM + 1e-9,
              note="the underlying cannot fall below zero, so the gain is capped")
c.assert_true("a long call's loss is capped at the premium",
              lambda: min(m.payoff("long_call", K, PREM, s) for s in range(0, 201)) >= -PREM - 1e-9,
              note="you can never lose more than you paid")

c.check("intrinsic value of an in-the-money call",
        lambda: m.intrinsic("call", 45, 48), 3.0, note="max(spot - strike, 0)")
c.check("intrinsic value of an out-of-the-money call",
        lambda: m.intrinsic("call", 55, 48), 0.0, note="never negative")
c.check("intrinsic value of an in-the-money put",
        lambda: m.intrinsic("put", 55, 52), 3.0, note="max(strike - spot, 0)")
c.check("extrinsic value from a quoted premium",
        lambda: m.extrinsic("put", 55, 52, 4.20), 1.20, tol=0.001,
        note="premium minus intrinsic")

sys.exit(c.report())
