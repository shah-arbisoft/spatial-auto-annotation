"""Invariants for content-adaptive frame selection."""

import numpy as np
import pytest

from src.keyframes import (Segment, boundary_agreement, distance,
                           segment_sequence, sweep, thumbnail)


def frame(value, size=(48, 64), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full(size, float(value))
    if noise:
        a = a + rng.normal(0, noise, size)
    return np.clip(a, 0, 255).astype(np.uint8)


def ramp(value, size=(48, 64)):
    """A frame with structure, so mean subtraction cannot erase it."""
    a = np.linspace(0, value, size[1])[None, :].repeat(size[0], axis=0)
    return a.astype(np.uint8)


class TestThumbnail:
    def test_shape_is_fixed(self):
        for h, w in [(48, 64), (480, 640), (17, 23)]:
            assert thumbnail(frame(100, (h, w))).shape == (48, 64)

    def test_colour_is_accepted(self):
        rgb = np.dstack([frame(100)] * 3)
        assert thumbnail(rgb).shape == (48, 64)

    def test_mean_is_removed(self):
        assert thumbnail(frame(30)).mean() == pytest.approx(0.0, abs=1e-4)

    def test_global_brightness_shift_is_ignored(self):
        """An auto-exposure step must not read as a content change."""
        a, b = thumbnail(ramp(200)), thumbnail(np.clip(ramp(200) + 40, 0, 255))
        assert distance(a, b) < 1.0

    def test_content_change_is_not_ignored(self):
        a, b = thumbnail(ramp(200)), thumbnail(ramp(200)[:, ::-1])
        assert distance(a, b) > 10.0


class TestSegmentation:
    def test_empty_sequence(self):
        assert segment_sequence([], 5.0) == []

    def test_identical_frames_form_one_segment(self):
        th = [thumbnail(ramp(200)) for _ in range(20)]
        segs = segment_sequence(th, 5.0)
        assert len(segs) == 1
        assert len(segs[0]) == 20

    def test_every_frame_distinct_gives_no_compression(self):
        th = [thumbnail(ramp(20 * i)) for i in range(1, 12)]
        assert len(segment_sequence(th, 0.5)) == 11

    def test_segments_partition_the_sequence(self):
        rng = np.random.default_rng(3)
        th = [thumbnail((rng.random((48, 64)) * 255).astype(np.uint8))
              for _ in range(40)]
        for tau in (1.0, 5.0, 20.0, 100.0):
            segs = segment_sequence(th, tau)
            covered = [i for s in segs for i in s.frames]
            assert covered == list(range(40)), "frames lost or duplicated"

    def test_hard_cut_is_detected(self):
        th = [thumbnail(ramp(200))] * 10 + [thumbnail(ramp(200)[:, ::-1])] * 10
        segs = segment_sequence(th, 5.0)
        assert len(segs) == 2
        assert segs[1].start == 10

    def test_slow_drift_accumulates_into_a_boundary(self):
        """The property consecutive-frame differencing lacks.

        A bright patch slides one pixel per frame, as a slowly panning camera
        moves the scene. Every consecutive step is small, but the drift away
        from the anchor is not, so a boundary must still appear — the failure
        mode that leaves shot detection blind to a robot walking.
        """
        th = []
        for i in range(40):
            a = np.zeros((48, 64), dtype=np.uint8)
            a[10:30, i:i + 20] = 255
            th.append(thumbnail(a))

        step = max(distance(th[i], th[i + 1]) for i in range(39))
        drift = distance(th[0], th[-1])
        assert drift > 5 * step, "fixture does not actually accumulate"

        # a threshold above every single step still has to fire on the drift
        tau = 2 * step
        segs = segment_sequence(th, tau)
        assert len(segs) > 1
        assert all(distance(th[i], th[s.keyframe]) <= 2 * drift
                   for s in segs for i in s.frames)

    def test_keyframe_lies_inside_its_segment(self):
        rng = np.random.default_rng(7)
        th = [thumbnail((rng.random((48, 64)) * 255).astype(np.uint8))
              for _ in range(30)]
        for s in segment_sequence(th, 15.0):
            assert s.start <= s.keyframe <= s.end
            assert s.keyframe in s.frames

    def test_larger_tau_never_makes_more_segments(self):
        rng = np.random.default_rng(11)
        th = [thumbnail((rng.random((48, 64)) * 255).astype(np.uint8))
              for _ in range(60)]
        counts = [len(segment_sequence(th, t)) for t in (1, 5, 10, 20, 50)]
        assert counts == sorted(counts, reverse=True)

    def test_min_len_folds_transients_away(self):
        th = ([thumbnail(ramp(200))] * 8 + [thumbnail(ramp(200)[:, ::-1])]
              + [thumbnail(ramp(200))] * 8)
        assert len(segment_sequence(th, 5.0, min_len=1)) == 3
        assert all(len(s) >= 2
                   for s in segment_sequence(th, 5.0, min_len=2)[1:])


class TestReporting:
    def test_sweep_is_monotone_in_compression(self):
        rng = np.random.default_rng(5)
        th = [thumbnail((rng.random((48, 64)) * 255).astype(np.uint8))
              for _ in range(50)]
        rows = sweep(th, [1, 5, 10, 30])
        comps = [r["compression"] for r in rows]
        assert comps == sorted(comps)

    def test_boundary_agreement_perfect(self):
        segs = [Segment(0, 9, 0, list(range(10))),
                Segment(10, 19, 10, list(range(10, 20)))]
        a = boundary_agreement(segs, [0, 10], tol=2)
        assert a["precision"] == 1.0 and a["recall"] == 1.0

    def test_boundary_agreement_respects_tolerance(self):
        segs = [Segment(0, 9, 0, list(range(10))),
                Segment(10, 19, 10, list(range(10, 20)))]
        assert boundary_agreement(segs, [0, 15], tol=2)["recall"] == 0.0
        assert boundary_agreement(segs, [0, 15], tol=6)["recall"] == 1.0
