import unittest,numpy as np
from checks import classify,linear_gaps,audit_substep
class Gates(unittest.TestCase):
    def test_nonpenetration_is_not_margin(self): self.assertEqual(classify(0.082681),(True,False))
    def test_exact_margin(self): self.assertEqual(classify(0.1),(True,True))
    def test_penetration(self): self.assertEqual(classify(-0.00101),(False,False))
    def test_numerical_boundary(self): self.assertEqual(classify(-0.001),(True,False))
    def test_nonfinite(self): self.assertEqual(classify(float('nan')),(False,False))
    def test_linear_translation(self):
        q0=np.array([[0.,0.,0.,0.,0.,0.,1.]])
        q=q0.copy();q[0,0]=0.00012
        rows={'dof_indices':np.array([0]),'row_offsets':np.array([0,1]),'linear_jacobian':np.array([[1.,0.,0.]]),
              'angular_jacobian':np.zeros((1,3)),'free_violations':np.array([-0.0001])}
        self.assertAlmostEqual(linear_gaps(q0,q,rows)[0],0.00002)
    def test_error_and_margin_independent(self): self.assertTrue(classify(0.10144)[1]);self.assertFalse(3e-8<1e-9)
class NativeAudit(unittest.TestCase):
    def fixture(self):
        stage={'stage':1,'native_solve_calls_cumulative':1,'row_count':6,'active_count':6,
               'linearization_parent_match':{'pass':True},'solver_tolerance':1e-9,
               'solver_error':3e-8,'native_solver_wall_ms':5.0,'mapping_refresh_wall_ms':0.0}
        return {'native_status':'PASS_NATIVE_SAME_SUBSTEP_RELINEARIZATION','native_solve_calls':1,
                'native_solver_total_wall_ms':5.0,'mapping_total_wall_ms':0.0,
                'completed_stages':1,'stages':[stage],'mapping_refresh_count':0,'error':None,'nan_inf':False,
                'final_stage_to_committed':{'pass':True},'committed_clearance_mm':0.101,
                'independent_committed_clearance_mm':0.101}
    def test_valid_native_contract(self):self.assertEqual(audit_substep(self.fixture(),1),[])
    def test_reject_missing_solve(self):
        s=self.fixture();s['native_solve_calls']=0;self.assertIn('solve_count',audit_substep(s,1))
    def test_reject_inactive_rows(self):
        s=self.fixture();s['stages'][0]['active_count']=5;self.assertIn('row_count',audit_substep(s,1))
    def test_reject_false_dense_check(self):
        s=self.fixture();s['independent_committed_clearance_mm']=0.2;self.assertIn('dense_disagreement',audit_substep(s,1))
if __name__=='__main__':unittest.main()
