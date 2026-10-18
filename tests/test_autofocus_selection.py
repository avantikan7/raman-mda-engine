from unittest.mock import patch

import numpy as np
import pytest
from scipy import ndimage as ndi
from useq import MDAEvent, MDASequence

from raman_mda_engine.autofocus_selection import (
    FocusCandidate, find_empty_patch, AcquisitionFocusSelection,
)


def test_cell_patch_avoided_even_when_blurred():
    rng = np.random.default_rng(8)
    y, x = np.indices((256, 320))
    cell = (x - 85)**2 + (y - 128)**2 < 45**2
    for blur in (1, 8):
        fluorescence = ndi.gaussian_filter(cell.astype(float) * 100, blur)
        bf = 100 + rng.normal(0, .2, cell.shape) + fluorescence
        candidate = find_empty_patch(bf, [fluorescence])
        assert candidate.x > 160
        assert candidate.radius_px >= 20
        assert 24 < candidate.y < 232


@pytest.mark.parametrize("frame", [np.zeros((128, 128)), np.ones((128, 128)), np.full((128, 128), np.nan)])
def test_unusable_bf_is_rejected(frame):
    with pytest.raises(ValueError):
        find_empty_patch(frame)


def test_clearance_and_registration_are_required():
    bf = np.random.default_rng(0).normal(100, 1, (128, 128))
    with pytest.raises(ValueError, match="No empty autofocus patch"):
        find_empty_patch(bf, min_radius_px=100)
    with pytest.raises(ValueError, match="matching shapes"):
        find_empty_patch(bf, [np.zeros((100, 128))])



def test_acquired_channels_wait_for_stack_and_keep_nearest_baseline():
    seq = MDASequence(channels=["BF", "GFP", "RM"], z_plan={"relative": [-2,0,2]}, axis_order="tpcz")
    events = [e for e in seq if e.channel.config != "RM"]
    selection = AcquisitionFocusSelection()
    selection.request(events[0])
    with patch("raman_mda_engine.autofocus_selection.find_empty_patch", return_value=FocusCandidate(50,50,30)) as detect:
        for e in events[:-1]:
            assert selection.consume(e, np.full((128,128), e.index["z"]+1), 20) is None
        assert not detect.called
        result = selection.consume(events[-1], np.ones((128,128)), 20)
        assert result.radius_px == 30
        np.testing.assert_array_equal(detect.call_args.args[0], np.full((128,128),2))
        np.testing.assert_array_equal(detect.call_args.args[1][0], np.full((128,128),2))
        selection.request(events[0])
        assert selection.consume(events[0], np.zeros((128,128)), 20) is None
        assert detect.call_count == 1
    assert not selection.frames


def test_sparse_channels_and_positions_do_not_mix():
    seq = MDASequence(channels=["BF", dict(config="GFP", acquire_every=2)],
                      stage_positions=[(0,0),(100,0)], time_plan={"interval":1,"loops":2})
    selection = AcquisitionFocusSelection()
    with patch("raman_mda_engine.autofocus_selection.find_empty_patch", return_value=FocusCandidate(50,50,30)) as detect:
        for event in seq:
            selection.request(event)
            result = selection.consume(event, np.full((128,128),event.index["p"]+1),20)
            if result is not None:
                bf, fp = detect.call_args.args
                assert np.all(bf == event.index["p"]+1)
                assert len(fp) == (1 if event.index["t"] == 0 else 0)
        assert detect.call_count == 4


def test_fluorescence_only_and_insufficient_clearance_skip():
    seq = MDASequence(channels=["GFP"])
    event = next(iter(seq))
    selection = AcquisitionFocusSelection()
    selection.request(event)
    frame = np.random.default_rng(1).normal(10,1,(128,128))
    result = selection.consume(event, frame, 1000)
    assert isinstance(result, str)
    assert "No empty autofocus patch" in result
    selection.request(event)
    assert selection.consume(event, frame, 1000) is None


def test_channel_without_stack_is_not_waited_for_at_every_z():
    seq = MDASequence(channels=[dict(config="BF",do_stack=False), "GFP"],
                      z_plan={"relative":[-1,0,1]},axis_order="tpcz")
    selection = AcquisitionFocusSelection()
    events = list(seq)
    selection.request(events[0])
    with patch("raman_mda_engine.autofocus_selection.find_empty_patch", return_value=FocusCandidate(50,50,30)) as detect:
        results = [selection.consume(e,np.ones((128,128)),20) for e in events]
    assert sum(r is not None for r in results) == 1
    assert detect.call_count == 1
