"""Pure/mocked tests: these never call Sofa.Simulation.animate or train."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from mcr_sim.config import DEFAULT_CONFIG, InstrumentConfig
from mcr_sim.instrument_layout import NelsonInstrumentLayout
from mcr_sim.magnetics import distributed_magnetic_torques, total_torque, nelson_dipole_moment_a_m2
from mcr_sim.magnetic_controller import MagneticActionController, quantize_half_away_from_zero
from mcr_sim.vessel_bundle import BranchData, VesselBundle, TYPE_I_PARENTS
from mcr_sim.pathfinder import StEVEBruteForceBFSAdapter, CenterlineRandomAdapter
from mcr_sim.reward import StEVEReward
from mcr_sim.task import NavigationTask
from mcr_sim.rl_env import BackendState, MagneticNavigationEnv
from mcr_sim.diagnostics import StepDiagnostics


def bundle(seed=101):
    coords = {"aorta": [[0,0,0],[0,0,10],[0,0,20],[0,0,30]],
              "bct": [[0,0,10],[5,0,10],[10,0,10]],
              "rcca": [[10,0,10],[15,0,15],[20,0,20]],
              "rsa": [[10,0,10],[15,0,5],[20,0,0]],
              "lcca": [[0,0,20],[-5,0,22],[-10,0,25]],
              "lsa": [[0,0,30],[5,0,33],[10,0,36]]}
    branches = {}
    for name, points in coords.items():
        points = np.asarray(points, dtype=float)
        radii = np.ones(len(points))
        branches[name] = BranchData(name, points, radii, points*.001, radii*.001)
    return VesselBundle(seed, Path("/mock"), Path("/mock/mesh.stl"), Path("/mock/visual.stl"),
                        Path("/mock/graph.vtk"), Path("/mock/metadata.json"),
                        {"scale_diameter_array": [0.85]}, branches, TYPE_I_PARENTS,
                        np.zeros(3), np.zeros(3), np.array([0,0,1.]), {}, None)


class FakeBackend:
    def reset(self, vessel):
        self.position = np.array([0.,0.,.001])
        self.field = np.array([.01,.01,0.])
        self.inserted = .001
        self.calls = []
        return self.state()
    def state(self):
        return BackendState(np.stack([self.position, np.zeros(3)]), np.eye(3), self.field.copy(), self.inserted)
    def physics_substep(self, magnetic_field_world_t, insertion_delta_m):
        self.field = np.asarray(magnetic_field_world_t)
        self.position[2] += insertion_delta_m
        self.inserted += insertion_delta_m
        self.calls.append((self.field.copy(), insertion_delta_m))
        return StepDiagnostics()
    def close(self): pass


class TestParameters(unittest.TestCase):
    def test_legacy_environment_cannot_replace_lcp(self):
        from mcr_sim.mcr_simulator import Simulator
        with patch.dict("os.environ", {"MCR_CONSTRAINT_SOLVER":"generic", "MCR_CONSTRAINT_TOLERANCE":".1", "MCR_CONSTRAINT_MAX_IT":"2"}):
            sim=Simulator(FakeNode())
        self.assertEqual(sim.lcp_solver.kind,"LCPConstraintSolver")
        self.assertEqual(sim.lcp_solver.kwargs["maxIt"],2000)
        self.assertEqual(sim.lcp_solver.kwargs["tolerance"],1e-4)

    def test_physical_and_rl_contract(self):
        c = DEFAULT_CONFIG; c.validate()
        self.assertEqual((c.instrument.body_length_m,c.instrument.tip_length_m),(.5,.034))
        self.assertEqual((c.instrument.young_modulus_body_pa,c.instrument.young_modulus_tip_pa),(100e6,21e6))
        self.assertEqual((c.instrument.outer_diameter_m,c.instrument.inner_diameter_m),(.00133,.0008))
        self.assertEqual((c.instrument.discretization.body_beams,c.instrument.discretization.tip_beams),(100,34))
        self.assertEqual(c.instrument.discretization.mechanical_dofs,135)
        self.assertEqual(c.observation.dimension,41)
        self.assertAlmostEqual(c.sofa.rl_dt_s,.132)
        self.assertEqual((len(c.vessel_seeds),len(c.eval_vessel_seeds)),(10,2))
        self.assertFalse(set(c.vessel_seeds)&set(c.eval_vessel_seeds))
    def test_magnet_overlap(self):
        layout = NelsonInstrumentLayout()
        self.assertEqual(layout.magnetic_node_indices,(131,132,133,134))
        np.testing.assert_allclose([x.weight for x in layout.magnet_overlaps],[.25]*4,atol=1e-12)
        self.assertAlmostEqual(sum(x.overlap_m for x in layout.magnet_overlaps),.004)
    def test_nelson_moment_and_torque(self):
        from mcr_sim.mcr_magnet import Magnet
        c = DEFAULT_CONFIG.instrument
        original = Magnet(c.magnet_length_m,c.magnet_outer_diameter_m,c.magnet_inner_diameter_m,c.magnet_remanence_t)
        self.assertAlmostEqual(nelson_dipole_moment_a_m2(),original.dipole_moment)
        torques = distributed_magnetic_torques([0,.014,0],[[0,0,0,1]]*4,NelsonInstrumentLayout())
        np.testing.assert_allclose(total_torque(torques),[0,0,original.dipole_moment*.014],rtol=1e-12)
    def test_quantization(self):
        self.assertEqual(quantize_half_away_from_zero(.0005,.001),.001)
        self.assertEqual(quantize_half_away_from_zero(-.0005,.001),-.001)
    def test_action_22_and_limits(self):
        plan = MagneticActionController().plan([1,1,-1],[.01,.01,0],np.eye(3))
        self.assertEqual(plan.field_substeps_world_t.shape,(22,3))
        self.assertEqual(plan.insertion_substeps_m.shape,(22,))
        self.assertAlmostEqual(plan.insertion_substeps_m.sum(),-.004)
        self.assertLessEqual(plan.magnetic_rotation_command_deg,24+1e-10)
        np.testing.assert_allclose(np.linalg.norm(plan.field_substeps_world_t,axis=1),math.sqrt(2)*.01)
    def test_invalid_action(self):
        for action in ([np.nan,0,0],[2,0,0]):
            with self.assertRaises(ValueError): MagneticActionController().plan(action,[.01,.01,0],np.eye(3))


class TestNavigation(unittest.TestCase):
    def test_bfs_not_euclidean(self):
        graph = StEVEBruteForceBFSAdapter(bundle())
        start,target=np.array([0,0,0]),np.array([20,0,20])
        self.assertGreater(graph.path_length_mm(start,target),np.linalg.norm(target-start))
    def test_bfs_same_point(self):
        graph = StEVEBruteForceBFSAdapter(bundle())
        self.assertAlmostEqual(graph.path_length_mm([1,0,10],[1,0,10]),0)
    def test_target_on_legal_centerline(self):
        b=bundle(); sampler=CenterlineRandomAdapter(b,("lcca","rcca","rsa","lsa","bct"))
        for seed in range(15):
            target=sampler.reset(seed)
            self.assertTrue(any(np.array_equal(target.coordinates_mm,p) for p in b.branches[target.branch].coordinates_mm))
        np.testing.assert_array_equal(sampler.reset(42).coordinates_mm,sampler.reset(42).coordinates_mm)
    def test_reward_units(self):
        reward=StEVEReward(); reward.reset(100)
        value=reward.step(90,True)
        self.assertAlmostEqual(value.total,1+.001*10-.005)
    def test_termination_and_time_limit(self):
        task=NavigationTask(); task.reset()
        result=task.step([0,0,0],[0,0,.0029])
        self.assertEqual((result.terminated,result.truncated,result.success),(True,False,True))
        task.reset(); task.steps=149
        result=task.step([0,0,0],[0,0,.01])
        self.assertEqual((result.terminated,result.truncated),(False,True))
        self.assertEqual(NavigationTask("eval").max_steps,450)
    def test_simerror_not_success(self):
        result=NavigationTask().step([0,0,0],[0,0,0],simulation_error=True)
        self.assertEqual((result.terminated,result.truncated,result.success),(False,True,False))
    def test_reset_observation_and_substep_allocation(self):
        backend=FakeBackend(); env=MagneticNavigationEnv(backend,bundles=[bundle()])
        observation,info=env.reset(seed=4,options={"target_branch":"lcca"})
        self.assertEqual(observation.shape,(41,))
        self.assertTrue(env.observation_space.contains(observation))
        observation,reward,terminated,truncated,info=env.step([1,1,1])
        self.assertEqual(len(backend.calls),22)
        self.assertAlmostEqual(sum(x[1] for x in backend.calls),.004)
        self.assertEqual(info["physics_substeps_attempted"],22)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIsNone(info["contact_count"])
        self.assertIsNone(info["max_penetration_m"])
    def test_cache_scale_guard(self):
        b=bundle(); b=replace(b,metadata={"scale_diameter_array":[1.0]})
        with self.assertRaises(ValueError): MagneticNavigationEnv(FakeBackend(),bundles=[b])
    def test_eval_uses_heldout_seed(self):
        env=MagneticNavigationEnv(FakeBackend(),mode="eval",bundles=[bundle(101),bundle(1101)])
        self.assertEqual([b.seed for b in env.bundles],[1101])
    def test_error_stops_substeps(self):
        class Failure(FakeBackend):
            def physics_substep(self,*args):
                super().physics_substep(*args)
                return StepDiagnostics(simulation_error=True)
        backend=Failure(); env=MagneticNavigationEnv(backend,bundles=[bundle()])
        env.reset(seed=0,options={"target_branch":"lcca"})
        _,_,terminated,truncated,info=env.step([0,0,0])
        self.assertEqual(len(backend.calls),1)
        self.assertEqual((terminated,truncated),(False,True))
    def test_nan_state_is_reported(self):
        class Failure(FakeBackend):
            def state(self):
                value=super().state()
                if self.calls: value.catheter_tracking_m[0,0]=np.nan
                return value
        backend=Failure(); env=MagneticNavigationEnv(backend,bundles=[bundle()])
        env.reset(seed=0,options={"target_branch":"lcca"})
        obs,_,terminated,truncated,info=env.step([0,0,0])
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual((terminated,truncated),(False,True))
        self.assertTrue(info["simulation_error"])
        self.assertIsNotNone(info["state_error"])


class FakeData:
    def __init__(self,value): self.value=value
class FakeObject:
    def __init__(self,kind,kwargs):
        self.kind=kind; self.kwargs=kwargs
        for key,value in kwargs.items(): setattr(self,key,FakeData(value))
class FakeNode:
    def __init__(self): self.objects=[]; self.children=[]
    def addChild(self,name):
        child=FakeNode(); child.name=name; self.children.append(child); return child
    def addObject(self,kind,**kwargs):
        obj=FakeObject(kind,kwargs) if isinstance(kind,str) else kind
        self.objects.append(obj); return obj
class FakeController:
    def __init__(self,*args,**kwargs): pass


class TestComponentWiring(unittest.TestCase):
    def test_actual_builder_forwards_locked_parameters(self):
        from mcr_sim.sofa_simulation import create_sofa_scene
        fake=SimpleNamespace(Core=SimpleNamespace(Controller=FakeController))
        with patch.dict(sys.modules,{"Sofa":fake}), patch("mcr_sim.sofa_compat.load_plugins",return_value={"mocked":True}):
            handles=create_sofa_scene(FakeNode(),bundle())
        shape=handles.instrument.rest_shape.kwargs
        self.assertEqual(shape["densityOfBeams"],[100,34])
        self.assertEqual(shape["numEdgesCollis"],[100,68])
        self.assertEqual(shape["numEdges"],600)
        self.assertEqual(handles.instrument.RG.kwargs["nx"],135)
        self.assertEqual(handles.instrument.CFF.kwargs["indices"],[131,132,133,134])
        self.assertEqual(handles.instrument.IRC.kwargs["speed"],0)
        self.assertTrue(handles.instrument.IRC.kwargs["listening"])
        self.assertEqual(handles.root.dt,.006)
        self.assertEqual(handles.simulator.lcp_solver.kind,"LCPConstraintSolver")
        lcp=handles.simulator.lcp_solver.kwargs
        self.assertEqual({k:lcp[k] for k in ("mu","tolerance","maxIt","build_lcp")},
                         {"mu":.1,"tolerance":1e-4,"maxIt":2000,"build_lcp":False})
        kinds=[x.kind for x in handles.vessel.CollisionModel.objects]
        self.assertIn("TriangleCollisionModel",kinds)
        self.assertIn("LineCollisionModel",kinds)
        self.assertNotIn("PointCollisionModel",kinds)
        self.assertEqual(handles.vessel.CollisionModel.objects[0].kwargs["scale"],"0.001")
        self.assertEqual(handles.stages["G"]["status"],"NOT_RUN")


class TestRecurrentSAC(unittest.TestCase):
    def test_episode_safe_multi_env_and_truncation_mask(self):
        from mcr_sim.recurrent_sac import EpisodeSequenceReplay
        replay=EpisodeSequenceReplay(10000,4,3)
        for i in range(3):
            replay.add(np.full(41,i),np.zeros(3),0,np.full(41,i+1),False,i==2,0)
            replay.add(np.full(41,100+i),np.zeros(3),0,np.full(41,101+i),False,False,1)
        replay.add(np.full(41,9),np.zeros(3),0,np.full(41,10),True,False,0)
        sequence,_=replay._sequence(6)
        np.testing.assert_array_equal(sequence,np.full((3,41),9))
        sequence,_=replay._sequence(4)
        np.testing.assert_array_equal(sequence[:,0],[0,1,2])
        self.assertEqual(replay.data[4].terminated,False)
        self.assertEqual(replay.data[4].truncated,True)
    def test_time_limit_bootstraps_but_success_does_not(self):
        from mcr_sim.recurrent_sac import EpisodeSequenceReplay
        replay=EpisodeSequenceReplay(10000,1)
        replay.add(np.zeros(41),np.zeros(3),0,np.ones(41),False,True)
        self.assertEqual(float(replay.sample("cpu")["done"][0,0]),0)
        replay.data.clear()
        replay.add(np.zeros(41),np.zeros(3),0,np.ones(41),True,False)
        self.assertEqual(float(replay.sample("cpu")["done"][0,0]),1)

    def test_forward_gradient_hidden_and_checkpoint(self):
        import torch
        from mcr_sim.recurrent_sac import RecurrentSAC
        torch.set_num_threads(2); torch.manual_seed(0)
        agent=RecurrentSAC(DEFAULT_CONFIG.sac)
        self.assertEqual(agent.networks.encoder.lstm.hidden_size,900)
        self.assertEqual(agent.networks.encoder.lstm.num_layers,1)
        self.assertEqual([m.out_features for m in agent.networks.actor.trunk if isinstance(m,torch.nn.Linear)],[400,400,400,6])
        for network in (agent.networks.critics.q1,agent.networks.critics.q2):
            self.assertEqual([m.out_features for m in network if isinstance(m,torch.nn.Linear)],[400,400,400,1])
        action,hidden=agent.action_with_state(np.zeros(41))
        self.assertEqual(action.shape,(3,))
        self.assertEqual(tuple(hidden[0].shape),(1,1,900))
        batch={"observation":torch.randn(32,2,41),"next_observation":torch.randn(32,2,41),
               "action":torch.zeros(32,3),"reward":torch.zeros(32,1),"done":torch.zeros(32,1)}
        before=next(agent.networks.encoder.parameters()).detach().clone()
        losses=agent.update(batch)
        self.assertTrue(all(math.isfinite(v) for v in losses.values()))
        self.assertFalse(torch.equal(before,next(agent.networks.encoder.parameters())))
        checkpoint=agent.checkpoint(); agent.restore(checkpoint)
        self.assertIn("critic_optimizer",checkpoint)


if __name__ == "__main__": unittest.main()
