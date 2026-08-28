import unittest
from typing import List

from pylabrobot.legacy.plate_reading.standard import ImagingMode, Objective
from pylabrobot.pymmcore.backend import PymmcoreImagerBackend, StagePlateCalibration
from pylabrobot.resources.corning.plates import cor_96_wellplate_360uL_Fb


class TestStagePlateCalibration(unittest.TestCase):
  def setUp(self):
    self.plate = cor_96_wellplate_360uL_Fb(name="plate")

  def test_a1_maps_to_a1_center(self):
    cal = StagePlateCalibration(a1_center_um=(1000.0, 2000.0))
    self.assertEqual(cal.well_center_to_stage_um(self.plate, 0, 0), (1000.0, 2000.0))

  def test_well_pitch(self):
    cal = StagePlateCalibration(a1_center_um=(0.0, 0.0))
    # 96-well plate: 9 mm pitch. B2 is one row down, one column right of A1.
    x, y = cal.well_center_to_stage_um(self.plate, 1, 1)
    self.assertAlmostEqual(x, 9000.0)
    # PLR y decreases toward higher row indices; default y_sign=-1 flips it.
    self.assertAlmostEqual(y, 9000.0)

  def test_axis_signs(self):
    cal = StagePlateCalibration(a1_center_um=(0.0, 0.0), x_sign=-1, y_sign=1)
    x, y = cal.well_center_to_stage_um(self.plate, 1, 1)
    self.assertAlmostEqual(x, -9000.0)
    self.assertAlmostEqual(y, -9000.0)


class _FakeEvents:
  def __init__(self):
    self._cbs: List = []

  def connect(self, cb):
    self._cbs.append(cb)

  def disconnect(self, cb):
    self._cbs.remove(cb)


class _FakeMDA:
  def __init__(self):
    self.events = type("E", (), {"frameReady": _FakeEvents()})()
    self.sequences: List = []

  def run(self, sequence):
    self.sequences.append(sequence)
    for i, event in enumerate(sequence):
      for cb in self.events.frameReady._cbs:
        cb(f"img{i}", event, {})

  def cancel(self):
    pass


class _FakeCore:
  def __init__(self):
    self.mda = _FakeMDA()
    self.properties: List = []

  def getAvailableConfigs(self, group):
    return ("DAPI", "phase-contrast")

  def getFocusDevice(self):
    return "Z"

  def getZPosition(self):
    return 42.0

  def setProperty(self, device, prop, value):
    self.properties.append((device, prop, value))


class TestPymmcoreImagerBackend(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.core = _FakeCore()
    self.plate = cor_96_wellplate_360uL_Fb(name="plate")
    self.backend = PymmcoreImagerBackend(
      core=self.core,
      channel_map={ImagingMode.DAPI: "DAPI", ImagingMode.PHASE_CONTRAST: "phase-contrast"},
      calibration=StagePlateCalibration(a1_center_um=(0.0, 0.0)),
      objective_labels={Objective.O_20X_PL_FL: "20x"},
      objective_device="Objective",
    )
    await self.backend.setup()

  async def test_capture_builds_sequence(self):
    result = await self.backend.capture(
      row=1,
      column=1,
      mode=ImagingMode.DAPI,
      objective=Objective.O_20X_PL_FL,
      exposure_time=50,
      focal_height=1.5,
      gain="machine-auto",
      plate=self.plate,
    )
    self.assertEqual(len(self.core.mda.sequences), 1)
    seq = self.core.mda.sequences[0]
    pos = seq.stage_positions[0]
    self.assertAlmostEqual(pos.x, 9000.0)
    self.assertAlmostEqual(pos.y, 9000.0)
    self.assertAlmostEqual(pos.z, 1500.0)  # mm -> um
    self.assertEqual(seq.channels[0].config, "DAPI")
    self.assertEqual(seq.channels[0].exposure, 50)
    self.assertEqual(result.images, ["img0"])
    self.assertEqual(self.core.properties, [("Objective", "Label", "20x")])

  async def test_capture_coverage_grid(self):
    result = await self.backend.capture(
      row=0,
      column=0,
      mode=ImagingMode.DAPI,
      objective=Objective.O_20X_PL_FL,
      exposure_time=10,
      focal_height=0.0,
      gain="machine-auto",
      plate=self.plate,
      coverage=(2, 3),
    )
    seq = self.core.mda.sequences[0]
    self.assertIsNotNone(seq.grid_plan)
    self.assertEqual(len(result.images), 6)

  async def test_unknown_channel_preset_raises_on_setup(self):
    backend = PymmcoreImagerBackend(
      core=_FakeCore(), channel_map={ImagingMode.GFP: "nonexistent"}
    )
    with self.assertRaises(ValueError):
      await backend.setup()

  async def test_auto_exposure_not_implemented(self):
    with self.assertRaises(NotImplementedError):
      await self.backend.capture(
        row=0,
        column=0,
        mode=ImagingMode.DAPI,
        objective=Objective.O_20X_PL_FL,
        exposure_time="machine-auto",
        focal_height=0.0,
        gain="machine-auto",
        plate=self.plate,
      )


if __name__ == "__main__":
  unittest.main()
