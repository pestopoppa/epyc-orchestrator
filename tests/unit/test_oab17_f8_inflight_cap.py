"""OAB-17 F8: concurrent scouted requests cannot jointly violate reserve_slots."""
from __future__ import annotations

from types import SimpleNamespace

from src.api.routes.chat_pipeline import scout_stage as ss


class _Resolver:
    def __init__(self, total, processing):
        self.occ = SimpleNamespace(slots=list(range(total)), processing=processing)

    def pool_occupancy(self, url):
        return self.occ


def _cap(url, resolver, hold):
    return ss.resolve_cap(url, max_scouts=8, n_targets=8, reserve_slots=1,
                          resolver=resolver, hold=hold)


def test_second_request_sees_the_first_ones_reservation():
    url = "http://127.0.0.1:59999"
    r = _Resolver(total=6, processing=0)            # free 6, reserve 1 -> 5 for the first
    h1, h2 = {}, {}
    assert _cap(url, r, h1)["cap"] == 5
    second = _cap(url, r, h2)                   # same stale snapshot
    assert second["cap"] == 0 and second["inflight_local"] == 5
    ss._release_inflight(h1["url"], h1["token"])
    assert _cap(url, r, {})["cap"] == 5          # released -> admitted again


def _inflight(url):
    with ss.scout_reservations.locked(url) as slate:
        return slate.inflight()


def test_no_hold_keeps_the_original_snapshot_behaviour():
    r = _Resolver(total=4, processing=1)
    assert ss.resolve_cap("http://x:1", max_scouts=8, n_targets=8, reserve_slots=1,
                          resolver=r)["cap"] == 2
    assert _inflight("http://x:1") == 0


def test_release_never_goes_negative():
    ss._release_inflight("http://y:1", "no-such-token")
    ss._release_inflight(None, "t")
    ss._release_inflight("http://y:1", None)
    assert _inflight("http://y:1") == 0
