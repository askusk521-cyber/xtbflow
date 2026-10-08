import numpy as np
from xtbflow.v1.xtb_chain import numerical_hessian


def test_central_difference_hessian():
    h=np.array([[3.,1.,0.],[1.,4.,2.],[0.,2.,5.]])
    actual=numerical_hessian(lambda x:h@x,np.array([.1,.2,.3]))
    np.testing.assert_allclose(actual,h,atol=1e-12)
    np.testing.assert_array_equal(actual,actual.T)
