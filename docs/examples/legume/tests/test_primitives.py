import unittest

import numpy as np
import legume.backend as bd
import legume


class TestGMEgrad(unittest.TestCase):
    '''
    Tests of various primitives
    '''
    def test_eigh(self):
        try:
            from autograd import grad
            legume.set_backend('autograd')
        except:
            return 0

        def obj(mat):
            # We make a Hermitian matrix using the upper triangular part of mat
            # as the real symmetric and the lower triangular part of mat as the
            # imaginary antisymmetric part
            mat_sym = bd.triu(mat) + bd.triu(mat, 1).transpose()
            mat_low = bd.triu(bd.transpose(mat), 1)
            mat_her = mat_sym + 1j * mat_low - 1j * bd.transpose(mat_low)
            w, v = bd.eigh(mat_her)
            return bd.sum(bd.abs(w)) + bd.sum(bd.abs(v))

        np.random.seed(0)
        N = 6
        mat = np.random.randn(N, N)
        gr_ag = grad(obj)(mat)

        # Numerical gradients
        gr_num = legume.utils.grad_num(obj, mat)

        diff = np.sum(np.abs(gr_num - gr_ag)**2) / np.linalg.norm(gr_num)
        self.assertLessEqual(diff, 1e-4)

    def test_eigsh(self):
        try:
            from autograd import grad
            legume.set_backend('autograd')
        except:
            return 0

        def obj(mat, k=3):
            # We make a Hermitian matrix using the upper triangular part of mat
            # as the real symmetric and the lower triangular part of mat as the
            # imaginary antisymmetric part
            mat_sym = bd.triu(mat) + bd.triu(mat, 1).transpose()
            mat_low = bd.triu(bd.transpose(mat), 1)
            mat_her = mat_sym + 1j * mat_low - 1j * bd.transpose(mat_low)
            w, v = bd.eigsh(mat_her, k=k)
            return bd.sum(bd.abs(w)) + bd.sum(bd.abs(v))

        np.random.seed(0)
        N = 6
        mat = np.random.randn(N, N)
        gr_ag = grad(obj)(mat)

        # Numerical gradients
        gr_num = legume.utils.grad_num(obj, mat)

        diff = np.sum(np.abs(gr_num - gr_ag)**2) / np.linalg.norm(gr_num)
        self.assertLessEqual(diff, 1e-4)


if __name__ == '__main__':
    unittest.main()
