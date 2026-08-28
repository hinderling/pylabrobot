"""A PLR Imager device exposing useq-schema acquisitions in PLR vocabulary.

``UseqImager`` extends the classic :class:`Imager` frontend (whose ``capture()`` API keeps
working unchanged) with two methods:

- :meth:`acquire`: run any :class:`useq.MDASequence` or iterable of ``MDAEvent`` as-is.
- :meth:`acquire_wells`: express a multi-dimensional acquisition in plate terms — PLR
  wells and :class:`ImagingMode` channels — and have it compiled to a ``MDASequence``
  (stage positions from the plate geometry + calibration, channel presets from the
  backend's channel map) and executed.

The intent is for these methods to eventually live on the ``Imager`` frontend proper,
with ``MDASequence`` as the interchange format any imaging backend can accept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

from pylabrobot.legacy.plate_reading.imager import Imager
from pylabrobot.legacy.plate_reading.standard import ImagingMode, Objective
from pylabrobot.pymmcore.backend import PymmcoreImagerBackend
from pylabrobot.resources.plate import Plate
from pylabrobot.resources.well import Well

try:
  import useq

  USE_USEQ = True
except ImportError as e:
  USE_USEQ = False
  _USEQ_IMPORT_ERROR = e

if TYPE_CHECKING:
  import numpy as np

WellIdentifier = Union[str, Well, Tuple[int, int]]


class UseqImager(Imager):
  """Imager whose backend executes useq-schema acquisitions (currently pymmcore-plus)."""

  backend: PymmcoreImagerBackend

  def _well_row_col(self, plate: Plate, well: WellIdentifier) -> Tuple[int, int]:
    """Resolve a well identifier to (row, column), respecting PLR's column-major item
    order (note: not ``divmod(idx, num_items_x)``, which mis-addresses non-square
    plates)."""
    if isinstance(well, tuple):
      return well
    if isinstance(well, str):
      well = plate.get_well(well)
    idx = plate.index_of_item(well)
    if idx is None:
      raise ValueError(f"Well {well} not in plate {plate.name}")
    column, row = divmod(idx, plate.num_items_y)
    return row, column

  def well_position(self, well: WellIdentifier, name: Optional[str] = None) -> "useq.Position":
    """The absolute stage position (useq.Position) of a well center of the loaded plate."""
    plate = self.get_plate()
    row, column = self._well_row_col(plate, well)
    x_um, y_um = self.backend.calibration.well_center_to_stage_um(plate, row, column)
    if name is None and isinstance(well, str):
      name = well
    return useq.Position(x=x_um, y=y_um, name=name)

  async def acquire(
    self,
    sequence: "Union[useq.MDASequence, Iterable[useq.MDAEvent]]",
    output: Optional[Any] = None,
  ) -> List[Tuple["np.ndarray", "useq.MDAEvent", dict]]:
    """Run a useq acquisition as-is. See :meth:`PymmcoreImagerBackend.acquire`."""
    return await self.backend.acquire(sequence, output=output)

  async def acquire_wells(
    self,
    wells: Sequence[WellIdentifier],
    channels: Dict[ImagingMode, float],
    objective: Optional[Objective] = None,
    focal_height: Optional[float] = None,
    z_plan: Optional[Any] = None,
    time_plan: Optional[Any] = None,
    grid_plan: Optional[Any] = None,
    axis_order: Optional[str] = None,
    output: Optional[Any] = None,
  ) -> List[Tuple["np.ndarray", "useq.MDAEvent", dict]]:
    """Acquire a multi-dimensional experiment over wells of the loaded plate.

    Args:
      wells: wells to visit, as names ("A1"), Well resources, or (row, column) tuples.
      channels: map of PLR imaging mode -> exposure time (ms). Each entry becomes a
        useq Channel using the backend's channel preset map.
      objective: optional PLR objective, applied before the sequence starts.
      focal_height: base focal height in mm for every position; z_plan offsets are
        relative to it. If None, relative z plans center on the current focus position.
      z_plan / time_plan / grid_plan: useq plans, passed through verbatim
        (e.g. ``useq.ZRangeAround(range=4, step=2)``, ``useq.TIntervalLoops(...)``).
      axis_order: useq axis order string (default useq's, "tpgcz").
      output: path or pymmcore-plus handler for streaming data to disk (OME-Zarr/TIFF).
    """
    if not USE_USEQ:
      raise RuntimeError(f"useq-schema is required: {_USEQ_IMPORT_ERROR}")
    unknown = [m for m in channels if m not in self.backend.channel_map]
    if unknown:
      raise ValueError(f"Imaging modes {unknown} have no channel preset in the backend.")

    if objective is not None:
      await self.backend._set_objective(objective)

    z_um = focal_height * 1000.0 if focal_height is not None else None
    positions = []
    for well in wells:
      pos = self.well_position(well)
      positions.append(pos.model_copy(update={"z": z_um}) if z_um is not None else pos)

    sequence = useq.MDASequence(
      stage_positions=positions,
      channels=[
        useq.Channel(config=self.backend.channel_map[mode], exposure=float(exposure))
        for mode, exposure in channels.items()
      ],
      z_plan=z_plan,
      time_plan=time_plan,
      grid_plan=grid_plan,
      **({"axis_order": axis_order} if axis_order is not None else {}),
    )
    return await self.backend.acquire(sequence, output=output)
