import unittest

import useq

from pylabrobot.legacy.plate_reading.standard import ImagingMode, Objective
from pylabrobot.pymmcore.backend import PymmcoreImagerBackend, StagePlateCalibration
from pylabrobot.pymmcore.backend_tests import _FakeCore
from pylabrobot.pymmcore.imager import UseqImager
from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.corning.plates import cor_96_wellplate_360uL_Fb


class TestUseqImager(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.core = _FakeCore()
    self.backend = PymmcoreImagerBackend(
      core=self.core,
      channel_map={ImagingMode.DAPI: "DAPI", ImagingMode.PHASE_CONTRAST: "phase-contrast"},
      calibration=StagePlateCalibration(a1_center_um=(0.0, 0.0)),
      objective_labels={Objective.O_20X_PL_FL: "20x"},
      objective_device="Objective",
    )
    self.imager = UseqImager(
      name="scope", size_x=10, size_y=10, size_z=10, backend=self.backend
    )
    self.plate = cor_96_wellplate_360uL_Fb(name="plate")
    self.imager.assign_child_resource(self.plate, location=Coordinate.zero())
    await self.imager.setup()

  async def test_well_position_by_name(self):
    pos = self.imager.well_position("B2")
    self.assertAlmostEqual(pos.x, 9000.0)
    self.assertAlmostEqual(pos.y, 9000.0)
    self.assertEqual(pos.name, "B2")

  async def test_well_position_by_well_object_column_major(self):
    # H1 is index 7 (column-major): row 7, column 0.
    pos = self.imager.well_position(self.plate.get_well("H1"))
    self.assertAlmostEqual(pos.x, 0.0)
    self.assertAlmostEqual(pos.y, 63000.0)

  async def test_acquire_wells_builds_sequence(self):
    frames = await self.imager.acquire_wells(
      wells=["A1", "B2"],
      channels={ImagingMode.PHASE_CONTRAST: 10, ImagingMode.DAPI: 50},
      objective=Objective.O_20X_PL_FL,
      focal_height=1.5,
      z_plan=useq.ZRangeAround(range=4, step=2),
    )
    seq = self.core.mda.sequences[0]
    self.assertEqual(len(seq.stage_positions), 2)
    self.assertEqual(seq.stage_positions[0].name, "A1")
    self.assertAlmostEqual(seq.stage_positions[1].x, 9000.0)
    self.assertAlmostEqual(seq.stage_positions[1].z, 1500.0)
    self.assertEqual([c.config for c in seq.channels], ["phase-contrast", "DAPI"])
    self.assertEqual(seq.z_plan, useq.ZRangeAround(range=4, step=2))
    # 2 wells x 2 channels x 3 z
    self.assertEqual(len(frames), 12)
    self.assertIn(("Objective", "Label", "20x"), self.core.properties)

  async def test_acquire_wells_unknown_mode_raises(self):
    with self.assertRaises(ValueError):
      await self.imager.acquire_wells(wells=["A1"], channels={ImagingMode.GFP: 10})

  async def test_capture_still_works(self):
    result = await self.imager.capture(
      well=(0, 0),
      mode=ImagingMode.DAPI,
      objective=Objective.O_20X_PL_FL,
      exposure_time=10,
      focal_height=0.0,
    )
    self.assertEqual(len(result.images), 1)


if __name__ == "__main__":
  unittest.main()
