from __future__ import annotations
import time
from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple
# from cns_control.autofocus import autofocus_w_raman
from pymmcore_plus.mda._engine import ImagePayload
import numpy as np
from loguru import logger
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent
from scipy.ndimage import center_of_mass
from qtpy.QtCore import QTimer
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
from ._error_handling import slack_notify
from ._events import QRamanSignaler as RamanSignaler
from .aiming import RamanAimingSource, SnappableRamanAimingSource
if TYPE_CHECKING:
    from mda_simulator import ImageGenerator
    from useq import MDASequence
class EventPayload(NamedTuple):
    image: np.ndarray
class fakeAcquirer:
    """For development."""
    def collect_spectra_relative(self, points, exposure=20):
        points = np.asarray(points)
        if points.min() < 0 or points.max() > 1:
            raise ValueError("Points must be in [0, 1]")
        if points.shape[1] != 2 or points.ndim != 2:
            raise ValueError(
                f"volts must have shape (N, 2) but has shape {points.shape}"
            )
        points = (np.ascontiguousarray(points) - 0.5) * 1.2
        return self.collect_spectra_volts(points, exposure)
    def collect_spectra_volts(self, points, exposure=20):
        points = np.ascontiguousarray(points)
        assert points.shape[1] == 2
        arr = np.random.randn(points.shape[0], 1340) * exposure
        return np.cumsum(arr, axis=1)
class RamanEngine(MDAEngine):
    _FILTER_DEVICE = "DigitalIO"
    _FILTER_MASK = 1 << 1  # Dev1/port0/line1
    def __init__(
        self,
        mmc: CMMCorePlus = None,
        default_rm_exp: float = 20.0,
        spectra_collector=None,
        transformer=None,
        max_volt=1.6,
        sources: list[RamanAimingSource] = None,
        batch=False,
        autofocus=True,
        autofocus_p=np.array([0]),
        autofocus_object='quartz',
        image_p=np.array([0]),
        segment_and_track=True,
        segment_channel='BF',
        shutter_device=None,
        segment_crop=True,
        cellpose_model='cyto2',
        tracking_config='particle_config.json',
        scale = 2,
        raman_glass_offset = 0.5,
        skip_imaging_for_same_pos = False,
        autofocus_search_range=60,
        search_pts=20,
        fine_search_range=1.5,
        fine_search_pts=15,
        refocus_every=1,
        image_x=1344,
        image_y=1024,
        config_file = None,
    ) -> None:
        """
        Create a pymmcore-plus mda engine that also collects Raman data.
        Parameters
        ----------
        mmc : CMMCorePlus
            The core to use, or None to use the current instance
        default_rm_exp : float
            The default raman exposure in ms. Used if nothing else provided.
        spectra_collector : SpectraCollector instance
            If None use the default - or nothign if not importable
        sources : iterable
            Collection of aiming sources to aim the raman laser.
        segment_channel : str
            Channel config to use when snapping the image to segment/track.
        shutter_device : str or None
            Name of the shutter to open/close around Raman collection. If
            None (default), no explicit shutter commands are issued -- the
            'RM' config is assumed to handle the shutter itself.
        autofocus_search_range : float
            +/- Z range (um) for the (coarse) autofocus scan.
        search_pts : int
            Number of Z planes in the coarse autofocus scan. Used by all
            autofocus objects (quartz/glass/cell/software/laser-coarse).
        fine_search_range : float
            +/- Z range (um) for the laser autofocus FINE scan.
        fine_search_pts : int
            Number of Z planes in the laser autofocus FINE scan.
        image_x : int
            X size of the camera image in pixels.
        image_y : int
            Y size of the camera image in pixels.
        """
        # MDAEngine uses the core immediately during its initialization (for
        # example, to read the auto-shutter state), so it must never receive
        # None. Resolve the singleton before calling the base constructor.
        mmc = mmc or CMMCorePlus.instance()
        super().__init__(mmc)
        self.raman_events = RamanSignaler()
        self._rng = np.random.default_rng()
        self._img_gen: ImageGenerator | None = None
        self._default_rm_exp = default_rm_exp
        self._max_volt = max_volt
        self.raman_events = RamanSignaler()
        self._spectra_collector = spectra_collector
        self._transformer = transformer
        self._daq = spectra_collector.daq
        self._batch = batch
        self._autofocus = autofocus
        self._autofocus_object = autofocus_object
        self._autofocus_p = autofocus_p
        self._image_p = image_p
        self._segment_and_track = segment_and_track
        self._segment_channel = segment_channel
        self._shutter_device = shutter_device
        self._segment_crop = segment_crop
        self._cellpose_model = cellpose_model
        self._tracking_config = tracking_config
        self._scale = scale
        self._raman_glass_offset = raman_glass_offset
        self._skip_imaging_for_same_pos = skip_imaging_for_same_pos
        self._last_operation_reloaded = False
        self._autofocus_search_range = autofocus_search_range
        self._laser_autofocus_active = False
        self._search_pts = search_pts
        self._fine_search_range = fine_search_range
        self._fine_search_pts = fine_search_pts
        self._refocus_every = max(1, int(refocus_every))
        self._config_file = config_file
        # image dimensions in pixels (X = 1344, Y = 1024 by default)
        self._image_x = image_x
        self._image_y = image_y
        self._config_file = config_file
        if self._spectra_collector is None:
            try:
                from raman_control import SpectraCollector
                self._spectra_collector = SpectraCollector.instance()
            except ImportError:
                self._spectra_collector = None
                logger.warning(
                    "Could not import SpectraCollector - No raman collection"
                )
        self._rm_meta = None
        self.aiming_sources = sources if sources is not None else []
        self._sources: list[RamanAimingSource]

        # ``super().__init__`` stored the resolved core in ``self._mmc``.

    @property
    def aiming_sources(self) -> list[RamanAimingSource]:
        return self._sources
    @aiming_sources.setter
    def aiming_sources(self, val: list[RamanAimingSource]):
        if val is None:
            self._sources = []
        elif all([isinstance(source, RamanAimingSource) for source in val]):
            self._sources = list(val)
        else:
            raise TypeError(
                "aiming_sources must be a list of objects"
                " conforming to the RamanAimingSource protocol."
            )
    @property
    def default_rm_exposure(self) -> Real:
        return self._default_rm_exp  # type: ignore
    @default_rm_exposure.setter
    def default_rm_exposure(self, val: Real):
        if not isinstance(val, Real):
            raise TypeError(
                f"default_rm_exposure must be a real number, got {type(val)}"
            )
        # ignore typing here because above is the best check
        # but mypy doesn't see float as part of Real so it's a mess
        self._default_rm_exp = val  # type: ignore
    def _sequence_axis_order(self, seq: MDASequence) -> tuple:
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)
    def _event_to_index(self, event: MDAEvent) -> tuple[int, ...]:
        return tuple(event.index[a] for a in self._axis_order)

    def _channel_group_for_event(self, event: MDAEvent) -> str:
        """Return the config group even when a sequence has no channel axis."""
        if event.channel is not None:
            return event.channel.group
        return self._mmc.getChannelGroup() or "Channel"

    def _set_focus_filter(self, inserted: bool) -> None:
        """Control port0/line1 through Micro-Manager's port-wide DigitalIO device."""
        if self._FILTER_DEVICE not in self._mmc.getLoadedDevices():
            if inserted:
                raise LookupError(
                    f"Cannot insert focus filter: "
                    f"{self._FILTER_DEVICE!r} is not loaded."
                )
            return

        current = int(self._mmc.getState(self._FILTER_DEVICE))
        desired = (
            current | self._FILTER_MASK
            if inserted
            else current & ~self._FILTER_MASK
        )

        if desired != current:
            self._mmc.setState(self._FILTER_DEVICE, desired)
            self._mmc.waitForDevice(self._FILTER_DEVICE)

    def record_raman(self, event: MDAEvent):
        """
        Record and save the raman spectra for the current position and time.
        Parameters
        ----------
        event : MDAEvent
                From the mda sequence.
        Returns
        -------
        spec : (N, 1340) array of float
        """
        def shrink_mask(pts):
            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            xs = np.unique(pts[:, 0])
            boolean = np.array([False]*pts.shape[0])
            for idx, pt in enumerate(pts):
                if (pt[0] == x_min):
                    if np.sum(pts[:, 0] == x_min) > 4:
                        y_min = np.min(pts[pts[:, 0] == x_min][:, 1])
                        y_max = np.max(pts[pts[:, 0] == x_min][:, 1])
                        if (pt[1] == y_min) or (pt[1] == y_max):
                            boolean[idx] = True
                    else:
                        boolean[idx] = True
                elif (pt[0] == x_max):
                    if np.sum(pts[:, 0] == x_max) > 4:
                        y_min = np.min(pts[pts[:, 0] == x_max][:, 1])
                        y_max = np.max(pts[pts[:, 0] == x_max][:, 1])
                        if (pt[1] == y_min) or (pt[1] == y_max):
                            boolean[idx] = True
                    else:
                        boolean[idx] = True
                else:
                    y_min = np.min(pts[pts[:, 0] == pt[0]][:, 1])
                    y_max = np.max(pts[pts[:, 0] == pt[0]][:, 1])
                    if (pt[1] == y_min) or (pt[1] == y_max):
                        boolean[idx] = True
            if len(pts[~boolean]) > 5:
                return pts[~boolean]
            else:
                return pts
        p, t = event.index["p"], event.index.get("t", 0)
        logger.info(f"collecting raman: {p=}, {t=}")
        points = []
        which = []
        for source in self.aiming_sources:
            new_points = source.get_mda_points(event)
            if 'cell' in source.name.lower():
                if self._batch and self._segment_and_track:
                    xy = (np.array(new_points)*[self._image_x, self._image_y])/self._scale
                    mask = self._last_segments[p]
                    cell_id = mask[*np.mean(xy, axis=0).astype(int)]
                    m = mask == cell_id
                    in_mask = m[xy[:, 0].astype(int), xy[:, 1].astype(int)] != 0
                    new_points = new_points[in_mask]
                    # new_points = shrink_mask(shrink_mask(shrink_mask(shrink_mask(shrink_mask(new_points)))))
                    new_points = shrink_mask(new_points)
                    ###########################
                points.append(new_points) # must be inside the loop
                which.extend([source.name] * len(new_points))
        points = np.vstack(points)

        # spec = self._spectra_collector.collect_spectra_relative(
        #     points, self._default_rm_exp
        # )
        # self._mmc.setConfig(event.channel.group, "RM")
        self.try_set_config(self._channel_group_for_event(event), "RM")
        volts = self._transformer.BF_to_volts((points*[self._image_x, self._image_y])[:, :]/[self._image_y, self._image_x], max_volts = self._max_volt)
        _ = self._spectra_collector.collect_spectra_pts(np.tile(volts[0], (2, 1)), 100)
        if not self._batch:
            spec = self._spectra_collector.collect_spectra_pts(
                volts, self._default_rm_exp
            )
        else:
            spec = self._spectra_collector.collect_spectra_pts_batch(
                volts, self._default_rm_exp
            )
        self.raman_events.ramanSpectraReady.emit(event, spec, points, which, self._default_rm_exp)
    @slack_notify
    def snap_raman(
        self,
        exposure: Real = None,
        aiming_sources: None
        | (SnappableRamanAimingSource | list[SnappableRamanAimingSource]) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Record raman.
        Parameters
        ----------
        exposure : real, optional
            The exposure time to use, defaults to the *default_rm_exposure*
        aiming_sources : list[SnappableAimingSource]
            The aiming sources to use
        Returns
        -------
        spec : (N, 1340) np.ndarray
        points : (N, 2) absolute positions in image space where laser was aimed
        which : (N,) label (e.g. 'cell' or 'bkd') for each point
        """
        points = []
        which = []
        if aiming_sources is None:
            aiming_sources = [
                source
                for source in self.aiming_sources
                if isinstance(source, SnappableRamanAimingSource)
            ]
        elif not isinstance(aiming_sources, list):
            aiming_sources = [aiming_sources]
        for source in aiming_sources:
            if not isinstance(source, SnappableRamanAimingSource):
                raise TypeError(
                    "All aiming sources must be SnappableRamanAimingSources"
                )
            new_points = source.get_current_points()
            points.append(new_points)
            which.extend([source.name] * len(new_points))
        points = np.vstack(points)
        if exposure is None:
            exposure = self._default_rm_exp  # type: ignore
        spec = self._spectra_collector.collect_spectra_relative(points, exposure)
        return spec, points, which
    @slack_notify
    def setup_sequence(self, sequence: MDASequence) -> None:
        super().setup_sequence(sequence)
        raman_meta = sequence.metadata.get("raman", None)
        if raman_meta:
            if self._spectra_collector is None:
                raise RuntimeError("Spectra Collector not set - cannot collect Raman.")
            if len(self.aiming_sources) == 0:
                raise RuntimeError("No aiming sources - cannot collect Raman.")
            self._rm_channel = raman_meta.get("channel", "BF")
            z = raman_meta.get("z", "all")
            z_index = self._sequence_axis_order(sequence).index("z")
            if isinstance(z, str):
                if z.lower() == "center":
                    n_z = sequence.shape[z_index]
                    if n_z % 2 == 0:
                        raise ValueError("for z=center n_z must be odd.")
                    z = np.array(n_z // 2)
                elif z.lower() in ["all", "stack"]:
                    z = np.arange(sequence.shape[z_index])
            else:
                z = np.asanyarray(z)
            self._rm_z = z
            self._rm_meta = raman_meta
        self._z_rel = sequence.z_plan.positions()
        self._last_segments = {}
        # gate on (t, pos) so seg-and-track fires once per position PER timepoint,
        # even with a single position. (-1, -1) is a sentinel that never matches.
        self._last_pos = (-1, -1)
        self._last_best_z = {}
        self._last_images = {}
        self._tracks = {}
    def update_aim(self, pos, event, img, use_same_img=False):
        # Segmentation and tracking are optional. Import them only when this
        # feature is enabled so importing/using the core engine with napari does
        # not initialize Cellpose, BTrack, or PyTorch.
        from raman_mda_engine.aiming.autotracking import (
            segment_single_img,
            track_one_T,
            update_pos_points,
        )

        print('-----------updating aim-----------')
        points = []
        which = []
        for source in self.aiming_sources:
            # if ("autofocus" in source.name.lower()) and (self._autofocus_object in ['glass', 'quartz']):
            if "autofocus" in source.name.lower():
                continue
            new_points = source.get_mda_points(event, transform=False) * [self._image_x, self._image_y]
            points.append(new_points)
            which.extend([source.name] * len(new_points))
        points = np.vstack(points)
        which = np.asarray(which)
        P = event.index.get("p", -1)
        T = event.index.get("t", -1)
        # Always segment the supplied image. The caller (setup_event) takes a
        # fresh BF snap right before every call, so a fresh segmentation is the
        # correct default. `use_same_img` only controls whether we *reuse* a
        # neighbouring position's mask instead of this one.
        new_mask = segment_single_img(
            img, self._scale,
            crop=self._segment_crop,
            cellpose_model=self._cellpose_model,
        )
        # Decide whether reusing the previous position's mask is even possible.
        can_reuse = (
            use_same_img
            and (P - 1) in self._last_segments
        )
        if T == 0:
            if can_reuse:
                self._last_segments[P] = self._last_segments[P - 1]
            else:
                self._last_segments[P] = new_mask
            self.raman_events.aimUpdated.emit(event, img, self._last_segments[P], points, points)
            return
        prev = self._last_segments[P]
        if can_reuse:
            labels = np.vstack([prev[None, :], self._last_segments[P-1][None, :]])
            tracked, new_pts = track_one_T(
                labels, self._scale, points,
                tracked=self._tracks.get(P - 1), use_same_img=True,
                tracking_config=self._tracking_config,
            )
        else:
            labels = np.vstack([prev[None, :], new_mask[None, :]])
            tracked, new_pts = track_one_T(
                labels, self._scale, points, use_same_img=False,
                tracking_config=self._tracking_config,
            )
        self._tracks[P] = tracked
        new_pts = np.array(new_pts)
        # Ensure new_pts has at least shape (2, 2)
        if new_pts.ndim < 2 or new_pts.shape[0] < 2 or new_pts.shape[1] < 2:
            print(f"[WARNING] new_pts shape {new_pts.shape} is too small, recomputing from original points")
            new_pts = points.copy()  # fall back to untransformed original points
        if can_reuse:
            self._last_segments[P] = self._last_segments[P-1]
        else:
            self._last_segments[P] = new_mask
        self.raman_events.aimUpdated.emit(event, img, self._last_segments[P], new_pts, points)
        for source in self.aiming_sources:
            # if ("autofocus" in source.name.lower()) and (self._autofocus_object in ['glass', 'quartz']):
            if "autofocus" in source.name.lower():
                continue
            update_pos_points(pos, new_pts[which == source.name], source._points)
            return
    def reload(self, N=100):
        n = 0.1  # starting sleep time
        for attempt in range(N):
            if attempt == N-1:
                print('reach reloading maxiter')
            try:
                time.sleep(n)
                print('reloading config...')
                try:
                    self._mmc.events.channelGroupChanged.disconnect()
                except Exception as e:
                    None
                try:
                    self._mmc.events.configGroupChanged.disconnect()
                except Exception as e:
                    None
                try:
                    self._mmc.events.propertyChanged.disconnect()
                except Exception as e:
                    None
                try:
                    self._mmc.events.systemConfigurationLoaded.disconnect()
                except Exception as e:
                    None
                try:
                    self._mmc.events.configSet.disconnect()
                except Exception as e:
                    None
                self._mmc.unloadAllDevices()
                if self._config_file == None:
                    self._mmc.loadSystemConfiguration("test3.cfg")
                else:
                    self._mmc.loadSystemConfiguration(self._config_file)
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.unloadAllDevices())
                # QTimer.singleShot(0, lambda: self._mmc.loadSystemConfiguration("test3.cfg"))
                # self.try_set_config("Channel", "GFP")
                self.try_set_config("Channel", "BF")
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                self._mmc.waitForSystem()
                return  # success!
            except Exception as e:
                n += 1  # increase wait time and retry
    def try_set_XYPosition(self, x, y, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('set xy reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                self._mmc.setXYPosition(x, y)
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.setXYPosition(x, y))
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                return  # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_get_XYPosition(self, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('get xy reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                X, Y = self._mmc.getXYPosition()
                return  X, Y # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
    def try_set_ZPosition(self, z, N=2):
        n = 0  # starting sleep time
        self._last_operation_reloaded = False
        for attempt in range(N):
            if attempt == int(N/2):
                print('set z reach maxiter, reloading...')
                self.reload()
                self._last_operation_reloaded = True
            try:
                time.sleep(n)
                self._mmc.setZPosition(z)
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.setPosition(z))
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                return  # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_get_ZPosition(self, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('get z reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                Z = self._mmc.getPosition()
                return  Z # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
    def try_snap_image(self, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('snap image reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                self._mmc.snapImage()
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.snapImage())
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                # time.sleep(0.25)
                return  # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_snap(self, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('snap image reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                fig = self._mmc.snap()
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.snapImage())
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                # time.sleep(0.25)
                return  fig # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_get_image(self, N=2):
        n = 0  # starting sleep time
        self._last_operation_reloaded = False
        for attempt in range(N):
            if attempt == int(N/2):
                print('get image reach maxiter, reloading...')
                self.reload()
                self._last_operation_reloaded = True
            try:
                time.sleep(n)
                image = self._mmc.getImage()
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.snapImage())
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                # time.sleep(0.25)
                return image # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_set_config(self, channel, group, N=2):
        n = 0  # starting sleep time
        self._last_operation_reloaded = False
        for attempt in range(N):
            if attempt == int(N/2):
                print('set config reach maxiter, reloading...')
                self.reload()
                self._last_operation_reloaded = True
            try:
                time.sleep(n)
                self._mmc.setConfig(channel, group)
                # QTimer.singleShot(0, lambda: self._mmc.setConfig(channel, group))
                # self._mmc.setExposure(10)
                self._mmc.waitForSystem()
                self._set_focus_filter(
                    self._laser_autofocus_active and group == "RM"
                )
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                # time.sleep(0.25)
                return  # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_setShutter(self, shutter, boolean, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('set shutter reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                self._mmc.setShutterOpen(shutter, boolean)
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.setShutterOpen(shutter, boolean))
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                # time.sleep(0.25)
                return  # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def try_setExp(self, exp, N=2):
        n = 0  # starting sleep time
        for attempt in range(N):
            if attempt == int(N/2):
                print('set exposure time reach maxiter, reloading...')
                self.reload()
            try:
                time.sleep(n)
                self._mmc.setExposure(exp)
                self._mmc.waitForSystem()
                # QTimer.singleShot(0, lambda: self._mmc.setShutterOpen(shutter, boolean))
                # QTimer.singleShot(0, lambda: self._mmc.waitForSystem())
                # time.sleep(0.25)
                return  # success!
            except RuntimeError:
                n += 1  # increase wait time and retry
    def software_autofocus(self, stack):
        """
        stack: numpy array of shape (N, X, Y)
        returns: index of best focus, scores for all planes
        """
        # def focus_measure(image):
        #     """Variance of Laplacian focus metric."""
        #     return cv2.Laplacian(image, cv2.CV_64F).var()
        def focus_measure(image):
            """Variance of Laplacian focus metric."""
            lap = cv2.Laplacian(image, cv2.CV_64F).flatten()
            q = 0.15
            low = np.quantile(lap, q)
            high = np.quantile(lap, 1-q)
            return lap[(lap > low) & (lap < high)].var()
        scores = [focus_measure(stack[i]) for i in range(stack.shape[0])]
        best_index = int(np.argmax(scores))
        return best_index, scores

    def autofocus_w_raman(self, last_z, pt, t, p, max_volt=1.6):
        focusZ = self.try_get_ZPosition()
        object = self._autofocus_object
        # NOTE: search_pts is now a single class-level argument (self._search_pts)
        # used by every autofocus object for the coarse scan. (Previously quartz
        # used 30 and the rest 20; set search_pts=30 on the engine if you want the
        # old quartz density.)
        if object=='quartz':
            start=100
            end=380
            search_range=self._autofocus_search_range
            search_pts=self._search_pts
        elif object=='glass':
            start=1150
            end=1650
            search_range=self._autofocus_search_range
            search_pts=self._search_pts
        elif object=='cell':
            start=1300
            end=1370
            search_range=self._autofocus_search_range
            search_pts=self._search_pts
        elif object=='software':
            search_range=self._autofocus_search_range
            search_pts=self._search_pts
        elif object=='laser':
            search_range=self._autofocus_search_range
            search_pts=self._search_pts
        if object in ['cell', 'glass', 'quartz']:
            self._daq.galvo.stop()
            self._mmc.stopSequenceAcquisition()
            volts = self._transformer.BF_to_volts((pt.reshape(1, -1)*[self._image_x, self._image_y])/[self._image_y, self._image_x], max_volts=self._max_volt)
            # if object in ['glass', 'quartz']:
            #     # volts = np.array([0, 0])
            #     volts = np.array([[0,0], [0,0]])
            self.try_set_config("Channel", "RM")
            # self.try_setShutter("Fluoshutter", True)

            coarse_Z = np.linspace(-search_range, search_range, search_pts)
            coarse_raman = []
            something_broke = 0
            for z in tqdm(coarse_Z):
                self.try_set_ZPosition(focusZ+z)
                if self._last_operation_reloaded:
                    something_broke += 1
                if object != 'cell':
                    spec = self._spectra_collector.collect_spectra_pts(np.array([volts[0], volts[0]]), 200)
                else:
                    spec = self._spectra_collector.collect_spectra_pts(np.array([volts[0], volts[0]]), 1000)
                coarse_raman.append(np.mean(spec[:, :], axis=0))
            coarse_raman = np.asarray(coarse_raman)
            if object == 'cell':
                cell_raman = np.median(coarse_raman[:, start:end], axis=1)
                interp_func = interp1d(coarse_Z, cell_raman, kind='cubic')
                # Finer x-values for interpolation
                x_fine = np.linspace(coarse_Z.min(), coarse_Z.max(), 200)
                y_fine = interp_func(x_fine)
                # Find the maximum
                max_index = np.argmax(y_fine)
                x_peak = x_fine[max_index]
                # y_peak = y_fine[max_index]
                max_laser_offset = x_peak
            elif object in ['glass', 'quartz']:
                def normalized_tanh(x, x0, width):
                    return 0.5 * (np.tanh((x - x0) / width) + 1)
                def rescale(data):
                    return (data - np.min(data)) / (np.max(data) - np.min(data))
                cell_raman = rescale(np.median(coarse_raman[:, start:end], axis=1) / np.median(coarse_raman))
                # interp_func = interp1d(coarse_Z, cell_raman, kind='cubic')
                # y_fine = interp_func(x_fine)
                x_fine = np.linspace(coarse_Z.min(), coarse_Z.max(), 200)
                interp_func_glass = interp1d(coarse_Z, cell_raman, kind='cubic')
                y_fine_glass = interp_func_glass(x_fine)
                # max_index = np.argmax(y_fine)
                # x_peak = x_fine[max_index]
                # # y_peak = y_fine[max_index]
                # max_laser_offset = x_peak
                # print(focusZ + x_fine[np.argmin(np.abs(y_fine_glass - 0.5))])
                # popt, _ = curve_fit(normalized︃_tanh, coarse_Z, cell_raman, p0=[0, 2], method='trf')
                # max_laser_offset = popt[0]-self._raman_glass_offset # 2 is the default offset between rm cell and rm glass, change accordingly
                max_laser_offset = x_fine[np.argmin(np.abs(y_fine_glass - 0.5))] - self._raman_glass_offset
        elif object == 'software':
            coarse_Z = np.linspace(-search_range, search_range, search_pts)
            figs = []
            something_broke = 0
            for z in tqdm(coarse_Z):
                self.try_set_ZPosition(focusZ+z)
                self.try_set_config("Channel", "BF")
                self.try_setExp(10)
                fig = self.try_snap()
                figs.append(fig)
                if self._last_operation_reloaded:
                    something_broke += 1
            figs = np.asarray(figs)
            _, scores = self.software_autofocus(figs)
            def rescale(data):
                return (data - np.min(data)) / (np.max(data) - np.min(data))
            scores = rescale(scores)
            interp_func = interp1d(coarse_Z, scores, kind='cubic')
            x_fine = np.linspace(coarse_Z.min(), coarse_Z.max(), 200)
            y_fine = interp_func(x_fine)
            max_index = np.argmax(y_fine)
            x_peak = x_fine[max_index]
            # y_peak = y_fine[max_index]
            max_laser_offset = x_peak - self._raman_glass_offset
            coarse_raman = None
        elif object == 'laser':
            # --- coarse laser scan: uses class-level search_pts ---

            volts = self._transformer.BF_to_volts((pt.reshape(1, -1)*[self._image_x, self._image_y])/[self._image_y, self._image_x], max_volts=self._max_volt)
            # _ = self._spectra_collector.collect_spectra_pts(np.array([volts[0], volts[0]]), 100)
            coarse_Z = np.linspace(-search_range, search_range, search_pts)
            figs = []
            something_broke = 0

            self._laser_autofocus_active = True
            try:
                # try_set_config writes RM=1, then adds the filter bit,
                # producing port state 3.
                self.try_set_config("Channel", "RM")

                _ = self._spectra_collector.collect_spectra_pts(
                    np.array([volts[0], volts[0]]),
                    100,
                )
                for z in tqdm(coarse_Z):
                    self.try_set_ZPosition(focusZ+z)
                    self.try_setExp(0.1)
                    fig = self.try_snap()
                    figs.append(fig)
                    if self._last_operation_reloaded:
                        something_broke += 1
                figs = np.asarray(figs)
                # np.save(f'debug1/figs_{t}_{p}.npy', figs)
                # np.save(f'debug1/zs_{t}_{p}.npy', coarse_Z + focusZ)
                scores = figs.sum(axis=1)
                coarse_max = coarse_Z[np.argmax(np.max(scores, axis=1))] + focusZ
                # --- fine laser scan: uses dedicated fine_search_range / fine_search_pts ---
                fine_Z = np.linspace(-self._fine_search_range, self._fine_search_range, self._fine_search_pts)
                figs = []
                something_broke = 0
                for z in tqdm(fine_Z):
                    self.try_set_ZPosition(coarse_max+z)
                    # self.try_set_config("Channel", "RM")
                    # self.try_setExp(0.1)
                    fig = self.try_snap()
                    figs.append(fig)
                    if self._last_operation_reloaded:
                        something_broke += 1
                figs = np.asarray(figs)
                # np.save(f'debug2/figs_{t}_{p}.npy', figs)
                # np.save(f'debug2/zs_{t}_{p}.npy', fine_Z + coarse_max)
                scores = figs.sum(axis=1)
                max_laser_offset = fine_Z[np.argmax(np.max(scores, axis=1))] - focusZ + coarse_max - self._raman_glass_offset
                coarse_raman = None
            finally:
                self._laser_autofocus_active = False
                # Converts RM+filter (3) back to ordinary RM (1).
                self._set_focus_filter(False)
        # self.try_setShutter("Fluoshutter", False)
        if np.abs(max_laser_offset) >= 3*search_range or something_broke != 0:
            return focusZ, last_z, coarse_raman
        return focusZ, max_laser_offset+focusZ, coarse_raman


    def setup_event(self, event: MDAEvent) -> None:
        if event.x_pos is not None or event.y_pos is not None or event.z_pos is not None:
            x = event.x_pos if event.x_pos is not None else self._mmc.getXPosition()
            y = event.y_pos if event.y_pos is not None else self._mmc.getYPosition()
            self.try_set_XYPosition(x, y)
        if event.channel is not None:
            self.try_set_config(event.channel.group, event.channel.config)
        if self._autofocus or self._segment_and_track:
            pos = event.index.get("p", 0)
            t = event.index.get("t", 0)
            # fire once per (timepoint, position) -- robust to single position
            if (t, pos) != self._last_pos:
                self._last_pos = (t, pos)
                # Do we refresh focus / aiming on THIS timepoint?
                # t == 0 always runs (need a baseline mask + z per position);
                # after that, only every _refocus_every timepoints.
                do_refocus = (t == 0) or (t % self._refocus_every == 0)
                # ---- Baseline best-z for this position ----
                if self._last_best_z.get(pos) is None:
                    if pos in self._autofocus_p or not self._last_best_z:
                        # autofocus position (or the very first position):
                        # trust the sequence z as the starting point
                        base = event.z_pos
                        if base is None:
                            base = self.try_get_ZPosition()
                    else:
                        # non-autofocus position: inherit the most recent
                        # best-z (carries the last autofocus correction)
                        # instead of the stale sequence z / wherever the
                        # previous position's z-stack finished
                        base = self._last_best_z[max(self._last_best_z.keys())]
                    self._last_best_z[pos] = base
                    print(f"[baseline z] pos={pos}: {base}")
                    self.try_set_ZPosition(self._last_best_z[pos])
                else:
                    self.try_set_ZPosition(
                        self._last_best_z[pos] + self._raman_glass_offset
                    )
                # ---- Step 1: autofocus ----
                if self._autofocus and do_refocus and pos in self._autofocus_p:
                    points = None
                    for source in self.aiming_sources:
                        if 'autofocus' in source.name.lower():
                            points = source.get_mda_points(event, transform=False)
                    pt = np.array(points[0])
                    print('-----------Autofocusing-----------')
                    _, self._bestZ, _ = self.autofocus_w_raman(
                        self._last_best_z[pos], pt, t, pos
                    )
                    print(f'autofocus z = {self._bestZ}')
                    self._last_best_z[pos] = self._bestZ
                # ---- Step 2: aiming update ----
                if self._segment_and_track and do_refocus:
                    self.try_set_ZPosition(
                        self._last_best_z[pos] + self._raman_glass_offset
                    )
                    time.sleep(0.5)
                    self.try_set_config(self._channel_group_for_event(event), self._segment_channel)
                    self.try_setExp(10)
                    self.try_snap_image()
                    image = self.try_get_image()
                    if self._last_operation_reloaded and pos in self._last_images:
                        image = self._last_images[pos]
                    self._last_images[pos] = image
                    if self._autofocus:
                        use_same = pos not in self._autofocus_p
                    else:
                        use_same = False
                    self.update_aim(pos, event, image, use_same_img=use_same)
                    self.try_set_ZPosition(self._last_best_z[pos])
            # per-plane z: ALWAYS runs, even when event.z_pos is None,
            # so the stage can never silently stay at the previous
            # position's last plane
            if "z" in event.index:
                self.try_set_ZPosition(
                    self._last_best_z[pos] + self._z_rel[event.index["z"]]
                )
        elif event.z_pos is not None:
            self.try_set_ZPosition(event.z_pos)
        if event.exposure is not None:
            self._mmc.setExposure(event.exposure)

    def exec_event(self, event: MDAEvent) -> Any:
        channel = event.channel
        if self._rm_meta:
            if event.channel.config == self._rm_channel and event.index["z"] in self._rm_z:
                # self.try_setShutter("Fluoshutter", True)
                
                self.record_raman(event)
                # if shutter:
                    # self.try_setShutter(shutter, False)
                self.try_set_config(self._channel_group_for_event(event), "BF")
        if self._skip_imaging_for_same_pos:
            if event.index["p"] in self._image_p:
                self.try_snap_image()
                if (
                    (event.index["z"] == len(self._z_rel)-1)
                    and (event.channel.config != "BF")
                ):
                    self.try_set_config(channel.group, "BF")

                meta = self.get_frame_metadata(event)
                image = self.try_get_image()
                yield ImagePayload(image=image, event=event, metadata=meta)
        else:
            self.try_snap_image()
            if (
                (event.index["z"] == len(self._z_rel)-1)
                and (event.channel.config != "BF")
            ):
                self.try_set_config(channel.group, "BF")

            meta = self.get_frame_metadata(event)
            image = self.try_get_image()
            yield ImagePayload(image=image, event=event, metadata=meta)