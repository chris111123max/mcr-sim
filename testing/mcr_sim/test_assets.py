"""Read-only loader regression using existing adult assets, not new RL caches."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from mcr_sim.config import PYTHON_DIR
from mcr_sim.vessel_bundle import load_vessel_bundle, coordinate_consistency


class TestSavedAssetLoader(unittest.TestCase):
    def test_existing_cache_mm_to_m(self):
        root=PYTHON_DIR / "stEVE_anatomical_vessels/generated"
        for seed in (101,202,303):
            bundle=load_vessel_bundle(root / ("arch_type_I_seed_%d" % seed))
            result=coordinate_consistency(bundle)
            self.assertTrue(result["mm_to_m_exact"])
            self.assertTrue(result["all_branches_connected"])
            np.testing.assert_array_equal(bundle.insertion_point_m,bundle.insertion_point_mm*.001)
            self.assertEqual(bundle.metadata["scale_diameter_array"],[1.0])
    def test_content_hash_rejects_mixed_assets(self):
        source=PYTHON_DIR / "stEVE_anatomical_vessels/generated/arch_type_I_seed_101"
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"centerline_graph.vtk").write_bytes((source/"centerline_graph.vtk").read_bytes())
            # Test only content-integrity validation; these are explicit mock STL bytes.
            (root/"collision_inner.stl").write_bytes(b"mock collision fixture")
            (root/"visual_wall.stl").write_bytes(b"mock visual fixture")
            meta=json.loads((source/"metadata.json").read_text())
            meta["asset_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            (root/"metadata.json").write_text(json.dumps(meta))
            load_vessel_bundle(root)
            (root/"collision_inner.stl").write_bytes(b"wrong vessel fixture")
            with self.assertRaisesRegex(ValueError,"Mismatched vessel"):
                load_vessel_bundle(root)


if __name__ == "__main__": unittest.main()
