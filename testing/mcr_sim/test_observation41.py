"""Observation-only geometry/replay tests; no real SOFA integration."""
from dataclasses import replace
import unittest
import numpy as np
from mcr_sim.config import DEFAULT_CONFIG
from mcr_sim.observation import MagneticObservationBuilder, sample_polyline_m
from mcr_sim.pathfinder import StEVEBruteForceBFSAdapter
from mcr_sim.magnetic_controller import MagneticActionController, rotation_matrix_from_rotvec
from mcr_sim.rl_env import MagneticNavigationEnv
from test_unified import bundle, FakeBackend


class TestObservation41(unittest.TestCase):
    def fixture(self):
        b = bundle()
        branches = dict(b.branches)
        radii = np.array([1., 3., 5., 7.])
        branches['aorta'] = replace(branches['aorta'], radii_mm=radii, radii_m=radii * .001)
        b = replace(b, branches=branches)
        tip = np.array([.002, .003, .005])
        tracking = tip + np.array([[0,0,0], [0,0,-.020], [.040,0,-.020], [.080,0,-.020]])
        return b, dict(catheter_tracking_m=tracking, target_m=np.array([.020,0,.020]),
                       last_action=np.array([.2,-.3,.5]), field_world_t=np.array([.01,.01,0]),
                       tip_frame_world=rotation_matrix_from_rotvec([.4,-.2,.7]),
                       remaining_time_ratio=.7, inserted_length_m=.267,
                       tip_displacement_world_m=np.array([.003,-.001,.002]), last_insertion_m=.002)

    def build(self, b, values):
        graph = StEVEBruteForceBFSAdapter(b)
        path = graph.shortest_path_mm(values['catheter_tracking_m'][0] * 1000, values['target_m'] * 1000)
        builder = MagneticObservationBuilder(b)
        return builder.build(path=path, **values).flatten()

    def test_all_indices_and_true_geometry_radius(self):
        b, v = self.fixture(); o = self.build(b, v)
        c = DEFAULT_CONFIG.observation; frame = v['tip_frame_world']; tip = v['catheter_tracking_m'][0]
        expected = np.zeros(41)
        expected[0:3] = frame.T @ (v['field_world_t'] / np.linalg.norm(v['field_world_t']))
        # Current route is aorta -> bct -> rcca, with a corner, not the Euclidean shortcut.
        references = np.array([[0,0,.005], [.003,0,.010],
                               [.010+.005/np.sqrt(2),0,.010+.005/np.sqrt(2)],
                               [.020,0,.020], [.020,0,.020]])
        expected[3:18] = ((frame.T @ (references-tip).T).T / c.spatial_scale_m).ravel()
        expected[18] = (.005 + .010 + np.sqrt(2)*.010) / c.remaining_path_scale_m
        shape_offsets = np.array([[0,0,-.010],[.010,0,-.020],[.040,0,-.020]])
        expected[19:28] = ((frame.T @ shape_offsets.T).T / c.spatial_scale_m).ravel()
        expected[28:31] = [.7,.5,.002/c.radius_scale_m]
        expected[31:34] = frame.T @ (v['target_m']-tip) / c.target_scale_m
        expected[34:37] = v['last_action']
        expected[37:40] = frame.T @ v['tip_displacement_world_m'] / c.tip_displacement_scale_m
        expected[40] = .5
        self.assertEqual(o.shape, (41,)); self.assertEqual(o.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(o)))
        np.testing.assert_allclose(o, expected, atol=1e-7)
        self.assertFalse(hasattr(MagneticObservationBuilder(b), 'history'))
        # Changing actual catheter shape changes only the shape block, not route points.
        changed = dict(v); changed['catheter_tracking_m'] = tip + np.array([[0,0,0],[0,-.08,0]])
        other = self.build(b, changed)
        np.testing.assert_allclose(o[:19], other[:19]); self.assertFalse(np.allclose(o[19:28], other[19:28]))

    def test_translation_rotation_and_combined_invariance(self):
        b, v = self.fixture(); expected = self.build(b, v)
        q = rotation_matrix_from_rotvec([.8,-.6,.3])
        for rotation, shift in ((np.eye(3), np.array([.7,-1.2,.9])),
                                (q, np.zeros(3)), (q,np.array([.7,-1.2,.9]))):
            branches = {}
            for name, branch in b.branches.items():
                world = (rotation @ branch.coordinates_m.T).T + shift
                branches[name] = replace(branch, coordinates_m=world, coordinates_mm=world*1000)
            insertion = rotation @ b.insertion_point_m + shift
            transformed = replace(b, branches=branches, insertion_point_m=insertion,
                                  insertion_point_mm=insertion*1000,
                                  insertion_direction=rotation @ b.insertion_direction)
            values = dict(v)
            values['catheter_tracking_m'] = (rotation @ v['catheter_tracking_m'].T).T + shift
            values['target_m'] = rotation @ v['target_m'] + shift
            values['tip_frame_world'] = rotation @ v['tip_frame_world']
            values['field_world_t'] = rotation @ v['field_world_t']
            values['tip_displacement_world_m'] = rotation @ v['tip_displacement_world_m']
            np.testing.assert_allclose(self.build(transformed, values), expected, atol=1e-6, rtol=1e-6)

    def test_short_geometry_duplicate_points_and_endpoint_clamping(self):
        actual = sample_polyline_m([[1,2,3],[1,2,3],[1,2,3.02]], [0,.01,.03,.06])
        np.testing.assert_allclose(actual, [[1,2,3],[1,2,3.01],[1,2,3.02],[1,2,3.02]])
        np.testing.assert_allclose(sample_polyline_m([[1,2,3]], [.01,.03,.06]), [[1,2,3]]*3)

    def test_fixed_scales_clipping_and_no_nan(self):
        b,v = self.fixture()
        v.update(remaining_time_ratio=-1, inserted_length_m=10, last_insertion_m=-1,
                 tip_displacement_world_m=np.ones(3)*100, last_action=np.ones(3)*2)
        o=self.build(b,v)
        self.assertTrue(np.all(np.isfinite(o))); self.assertTrue(np.all(np.abs(o)<=1))
        self.assertEqual(o[28],0); self.assertEqual(o[29],1); self.assertEqual(o[40],-1)
        _, valid = self.fixture()
        valid_path = StEVEBruteForceBFSAdapter(b).shortest_path_mm(valid['catheter_tracking_m'][0]*1000, valid['target_m']*1000)
        for key in ('catheter_tracking_m','target_m','last_action','field_world_t','tip_frame_world',
                    'remaining_time_ratio','inserted_length_m','tip_displacement_world_m','last_insertion_m'):
            for bad in (np.nan,np.inf):
                _,good = self.fixture(); values=dict(good)
                values[key] = np.full_like(np.asarray(good[key],dtype=float),bad)
                with self.subTest(key=key,bad=bad), self.assertRaises(ValueError):
                    MagneticObservationBuilder(b).build(path=valid_path, **values)

    def test_radius_comes_from_vessel_not_constant(self):
        b,v=self.fixture(); original=self.build(b,v)
        doubled=replace(b,branches={name:replace(branch,radii_mm=branch.radii_mm*2,radii_m=branch.radii_m*2)
                                   for name,branch in b.branches.items()})
        changed=self.build(doubled,v)
        self.assertAlmostEqual(changed[30],original[30]*2)
        np.testing.assert_allclose(np.delete(changed,30),np.delete(original,30))

    def test_environment_actual_motion_action_time_and_reward(self):
        class ClampedBackend(FakeBackend):
            def physics_substep(self, magnetic_field_world_t, insertion_delta_m):
                return super().physics_substep(magnetic_field_world_t, insertion_delta_m*.25)
        backend=ClampedBackend(); env=MagneticNavigationEnv(backend,bundles=[bundle()])
        reset,initial=env.reset(seed=4,options={'target_branch':'lcca'})
        np.testing.assert_array_equal(reset[34:41],np.zeros(7)); self.assertEqual(reset[28],1)
        action=np.array([1.,1.,1.]); plan=MagneticActionController().plan(action,backend.field,np.eye(3))
        o,reward,_,_,info=env.step(action)
        np.testing.assert_allclose(o[34:36],plan.executed_action[:2],atol=1e-7)
        self.assertAlmostEqual(o[36],.25); self.assertAlmostEqual(o[40],.25)
        np.testing.assert_allclose(o[37:40],[0,0,.1],atol=1e-7)
        self.assertAlmostEqual(float(o[28]),149/150,places=6)
        self.assertAlmostEqual(float(o[29]),.002/.534,places=7)
        self.assertEqual(len(backend.calls),22)
        np.testing.assert_allclose(np.array([x[0] for x in backend.calls]),plan.field_substeps_world_t)
        self.assertAlmostEqual(info['requested_insertion_m'],.004)
        self.assertAlmostEqual(info['executed_insertion_m'],.001)
        expected=(float(info['success']) + .001*(initial['path_length_mm']-info['path_length_mm'])-.005)
        self.assertAlmostEqual(reward,expected)
        reset,_=env.reset(seed=4,options={'target_branch':'lcca'})
        np.testing.assert_array_equal(reset[34:41],np.zeros(7)); self.assertEqual(reset[28],1)
        evaluation=MagneticNavigationEnv(FakeBackend(),mode='eval',bundles=[bundle(1101)])
        evaluation.reset(seed=4,options={'target_branch':'lcca'})
        observed,*_=evaluation.step([0,0,0]); self.assertAlmostEqual(float(observed[28]),449/450,places=6)


class TestReplay41(unittest.TestCase):
    def test_legacy_dimensions_and_nonfinite_are_rejected(self):
        from mcr_sim.recurrent_sac import EpisodeSequenceReplay
        replay=EpisodeSequenceReplay(10000,1)
        for size in (20,27,40,42):
            with self.assertRaises(ValueError): replay.add(np.zeros(size),np.zeros(3),0,np.zeros(41),False)
            with self.assertRaises(ValueError): replay.add(np.zeros(41),np.zeros(3),0,np.zeros(size),False)
        for bad in (np.nan,np.inf):
            with self.assertRaises(ValueError): replay.add(np.full(41,bad),np.zeros(3),0,np.zeros(41),False)
        self.assertEqual(len(replay),0)

    def test_replay_lstm_input_and_online_memory(self):
        import torch
        from mcr_sim.recurrent_sac import EpisodeSequenceReplay
        from mcr_sim.networks import SACNetworks
        torch.set_num_threads(2)
        replay=EpisodeSequenceReplay(10000,32)
        for i in range(32): replay.add(np.zeros(41),np.zeros(3),0,np.ones(41),False)
        batch=replay.sample('cpu')
        self.assertEqual(tuple(batch['observation'].shape),(32,8,41))
        self.assertEqual(tuple(batch['next_observation'].shape),(32,8,41))
        network=SACNetworks(); self.assertEqual(network.encoder.lstm.input_size,41)
        with torch.no_grad():
            action,logp=network.policy(batch['observation'])
            self.assertEqual(tuple(action.shape),(32,3)); self.assertEqual(tuple(logp.shape),(32,1))
            _,h1=network.encoder.step(torch.zeros(41))
            _,h2=network.encoder.step(torch.zeros(41),h1)
        self.assertFalse(torch.equal(h1[0],h2[0]))
        for size in (20,27):
            with self.assertRaises(ValueError): network.encoder(torch.zeros(1,8,size))
            with self.assertRaises(ValueError): network.encoder.step(torch.zeros(size))


if __name__=='__main__': unittest.main()
