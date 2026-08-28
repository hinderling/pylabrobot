"""Imager backend driving any Micro-Manager-supported microscope via pymmcore-plus.

This backend translates PLR's imaging API into a :class:`useq.MDASequence` and executes
it with pymmcore-plus's MDA runner. Any microscope with a Micro-Manager device adapter
(or a pure-python ``UniMMCore`` device) can therefore be used as a PLR ``Imager``.

Coordinate model: PLR addresses wells as ``(row, column)`` on a :class:`Plate` resource;
Micro-Manager uses absolute stage coordinates in um. The bridge is
:class:`StagePlateCalibration`: the stage coordinate of well A1's center, plus axis
direction signs. Well offsets are computed from the PLR plate geometry (mm), so any
plate in the PLR resource library works without further configuration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from pylabrobot.legacy.plate_reading.backend import ImagerBackend
from pylabrobot.legacy.plate_reading.standard import (
  Exposure,
  FocalPosition,
  Gain,
  ImagingMode,
  ImagingResult,
  Objective,
)
from pylabrobot.resources.plate import Plate

try:
  import useq

  USE_USEQ = True
except ImportError as e:
  USE_USEQ = False
  _USEQ_IMPORT_ERROR = e

try:
  from pymmcore_plus import CMMCorePlus

  USE_PYMMCORE = True
except ImportError as e:
  USE_PYMMCORE = False
  _PYMMCORE_IMPORT_ERROR = e

if TYPE_CHECKING:
  import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StagePlateCalibration:
  """Maps PLR plate coordinates to absolute stage coordinates.

  Attributes:
    a1_center_um: stage position (x, y) in um at which the center of well A1 is in the
      optical axis.
    x_sign: +1 if increasing PLR x (toward higher column numbers) corresponds to
      increasing stage x, else -1.
    y_sign: +1 if increasing PLR y (toward row A / the back of the plate) corresponds to
      increasing stage y, else -1. Most inverted-microscope stages use -1.
  """

  a1_center_um: Tuple[float, float] = (0.0, 0.0)
  x_sign: int = 1
  y_sign: int = -1

  def well_center_to_stage_um(self, plate: Plate, row: int, column: int) -> Tuple[float, float]:
    """Compute the absolute stage position (um) of a well center."""
    a1 = plate.get_item((0, 0))
    well = plate.get_item((row, column))
    if a1.location is None or well.location is None:
      raise ValueError(f"Plate {plate.name} wells have no locations; cannot map to stage.")
    a1_center = a1.location + a1.center()
    well_center = well.location + well.center()
    dx_mm = well_center.x - a1_center.x
    dy_mm = well_center.y - a1_center.y
    return (
      self.a1_center_um[0] + self.x_sign * dx_mm * 1000.0,
      self.a1_center_um[1] + self.y_sign * dy_mm * 1000.0,
    )


class PymmcoreImagerBackend(ImagerBackend):
  """PLR imager backend backed by pymmcore-plus.

  ``capture()`` builds a single-position :class:`useq.MDASequence` and runs it on the
  core's MDA runner; ``acquire()`` accepts an arbitrary ``MDASequence`` directly for
  multi-dimensional acquisitions (z-stacks, time series, grids, multi-channel).

  Args:
    channel_map: maps PLR :class:`ImagingMode` to a Micro-Manager channel config preset
      name (in ``channel_group``).
    calibration: plate-to-stage coordinate calibration.
    core: an existing ``CMMCorePlus`` (or compatible, e.g. ``UniMMCore``) instance. If
      None, a new ``CMMCorePlus`` is created on setup.
    mm_config: path to a Micro-Manager system configuration file, loaded on setup when
      given.
    channel_group: name of the Micro-Manager config group holding the channel presets.
    objective_labels: optional map of PLR :class:`Objective` to a state-device label.
    objective_device: state device to apply ``objective_labels`` to (e.g.
      ``"Objective"``).
    gain_property: optional ``(device, property)`` to set from the ``gain`` argument.
  """

  def __init__(
    self,
    channel_map: Dict[ImagingMode, str],
    calibration: Optional[StagePlateCalibration] = None,
    core: Optional["CMMCorePlus"] = None,
    mm_config: Optional[str] = None,
    channel_group: str = "Channel",
    objective_labels: Optional[Dict[Objective, str]] = None,
    objective_device: Optional[str] = None,
    gain_property: Optional[Tuple[str, str]] = None,
  ):
    if not USE_USEQ:
      raise RuntimeError(
        f"useq-schema is required for PymmcoreImagerBackend. Import error: {_USEQ_IMPORT_ERROR}. "
        "Install with `pip install pylabrobot[pymmcore]`."
      )
    if not USE_PYMMCORE:
      raise RuntimeError(
        f"pymmcore-plus is required for PymmcoreImagerBackend. Import error: "
        f"{_PYMMCORE_IMPORT_ERROR}. Install with `pip install pylabrobot[pymmcore]`."
      )
    self.channel_map = channel_map
    self.calibration = calibration or StagePlateCalibration()
    self._core = core
    self.mm_config = mm_config
    self.channel_group = channel_group
    self.objective_labels = objective_labels or {}
    self.objective_device = objective_device
    self.gain_property = gain_property

  @property
  def core(self) -> "CMMCorePlus":
    if self._core is None:
      raise RuntimeError("Core not initialized. Run setup() first.")
    return self._core

  async def setup(self) -> None:
    if self._core is None:
      self._core = CMMCorePlus()
    if self.mm_config is not None:
      await asyncio.get_running_loop().run_in_executor(
        None, self._core.loadSystemConfiguration, self.mm_config
      )
    available = set(self._core.getAvailableConfigs(self.channel_group))
    unknown = {name for name in self.channel_map.values() if name not in available}
    if unknown:
      raise ValueError(
        f"Channel presets {sorted(unknown)} not defined in config group "
        f"'{self.channel_group}'. Available: {sorted(available)}"
      )

  async def stop(self) -> None:
    if self._core is not None:
      self._core.mda.cancel()

  async def acquire(
    self, sequence: "Union[useq.MDASequence, Iterable[useq.MDAEvent]]"
  ) -> List[Tuple["np.ndarray", "useq.MDAEvent", dict]]:
    """Run a useq MDASequence (or any iterable of MDAEvents) and return
    ``(image, event, metadata)`` frames.

    This is the native interface: anything expressible in useq-schema (z-stacks, time
    series, grids, channels, per-position sub-sequences, SLM images, hardware autofocus)
    runs as-is on pymmcore-plus's acquisition engine. Passing a generator or queue-backed
    iterable of events enables reactive ("smart microscopy") acquisitions where analysis
    of earlier frames decides later events.
    """
    frames: List[Tuple["np.ndarray", "useq.MDAEvent", dict]] = []

    def _on_frame(image: "np.ndarray", event: "useq.MDAEvent", metadata: dict) -> None:
      frames.append((image, event, metadata))

    self.core.mda.events.frameReady.connect(_on_frame)
    try:
      await asyncio.get_running_loop().run_in_executor(None, self.core.mda.run, sequence)
    finally:
      self.core.mda.events.frameReady.disconnect(_on_frame)
    return frames

  async def capture(
    self,
    row: int,
    column: int,
    mode: ImagingMode,
    objective: Objective,
    exposure_time: Exposure,
    focal_height: FocalPosition,
    gain: Gain,
    plate: Plate,
    coverage: Optional[Tuple[int, int]] = None,
    overlap: float = 0.0,
    **backend_kwargs: Any,
  ) -> ImagingResult:
    """Capture image(s) of a well by building and running a one-position MDASequence."""
    if mode not in self.channel_map:
      raise ValueError(f"ImagingMode {mode} has no channel preset in channel_map.")
    if isinstance(exposure_time, str):  # "machine-auto"
      raise NotImplementedError(
        "Auto exposure is not implemented in PymmcoreImagerBackend; pass a value in ms."
      )

    await self._set_objective(objective)
    await self._set_gain(gain)

    x_um, y_um = self.calibration.well_center_to_stage_um(plate, row, column)
    z_um: Optional[float]
    if isinstance(focal_height, (int, float)):
      z_um = float(focal_height) * 1000.0  # PLR mm -> stage um
    else:  # "machine-auto": keep the stage where it is
      z_um = self.core.getZPosition() if self.core.getFocusDevice() else None

    grid_plan = None
    if coverage is not None and coverage != (1, 1):
      rows, cols = coverage
      grid_plan = useq.GridRowsColumns(
        rows=rows, columns=cols, overlap=(overlap, overlap), relative_to="center"
      )

    sequence = useq.MDASequence(
      stage_positions=[useq.Position(x=x_um, y=y_um, z=z_um)],
      channels=[useq.Channel(config=self.channel_map[mode], exposure=float(exposure_time))],
      grid_plan=grid_plan,
    )

    frames = await self.acquire(sequence)
    return ImagingResult(
      images=[image for image, _, _ in frames],
      exposure_time=float(exposure_time),
      focal_height=(z_um / 1000.0) if z_um is not None else 0.0,
    )

  async def _set_objective(self, objective: Objective) -> None:
    if self.objective_device is None:
      return
    if objective not in self.objective_labels:
      raise ValueError(
        f"Objective {objective} has no label in objective_labels for device "
        f"'{self.objective_device}'."
      )
    label = self.objective_labels[objective]
    await asyncio.get_running_loop().run_in_executor(
      None, self.core.setProperty, self.objective_device, "Label", label
    )

  async def _set_gain(self, gain: Gain) -> None:
    if isinstance(gain, str):  # "machine-auto": leave camera at its current gain
      return
    if self.gain_property is None:
      logger.warning("gain=%s requested but no gain_property configured; ignoring.", gain)
      return
    device, prop = self.gain_property
    await asyncio.get_running_loop().run_in_executor(
      None, self.core.setProperty, device, prop, gain
    )
