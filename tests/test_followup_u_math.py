import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('umath',Path(__file__).parents[1]/'scripts/followup_u_math.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_water_rigid_span_and_positive_internal():
    x=np.array([[0.,0.,0.],[.95,0.,0.],[-.2,.9,0.]])
    q,rank=m.internal_basis(x)
    assert rank==6 and q.shape==(9,3)
    h=q@np.diag([2.,3.,4.])@q.T
    result=m.hessian_metrics([8,1,1],x,h,q[:,0])
    assert result['imaginary_count']==0
    assert min(result['frequencies_cm1'])>0
    assert abs(result['c_u']-2)<1e-12
    assert abs(result['overlap']-1)<1e-12


def test_negative_mode_and_translation_invariance():
    x=np.array([[0.,0.,0.],[.95,0.,0.],[-.2,.9,0.]])
    q,_=m.internal_basis(x);h=q@np.diag([-20.,30.,40.])@q.T
    result=m.hessian_metrics([8,1,1],x,h,q[:,0])
    shifted=m.hessian_metrics([8,1,1],x+3,h,q[:,0])
    assert result['imaginary_count']==1
    assert abs(result['best_neg_overlap']-1)<1e-12
    assert abs(result['c_u']-shifted['c_u'])<1e-12


def test_central_difference_and_symmetry():
    h=np.diag(np.arange(1.,10.))
    gradient=lambda x:(h@x.ravel()).reshape(-1,3)
    computed=m.finite_difference_hessian(np.ones((3,3)),gradient)
    assert np.allclose(computed,h)
    assert np.array_equal(computed,computed.T)
