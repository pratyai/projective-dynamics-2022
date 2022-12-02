import unittest
from constraints.spring import Spring
import helpers as hlp
import numpy as np
import numpy.linalg as la
import constants as const


class TestProjection(unittest.TestCase):
    def test_compressed(self):
        q0 = np.array([0, 0, 0])
        q1 = np.array([0.5, 0, 0])
        want_p = np.array([[-.25, 0, 0], [.75, 0, 0]])
        c = Spring(k=1, L=1, q=lambda: [(q0, None), (q1, None)])

        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p)

    def test_expanded(self):
        q0 = np.array([0, 0, 0])
        q1 = np.array([1.5, 0, 0])
        want_p = np.array([[.25, 0, 0], [1.25, 0, 0]])
        c = Spring(k=1, L=1, q=lambda: [(q0, None), (q1, None)])

        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p)

    def test_exact(self):
        q0 = np.array([0, 0, 0])
        q1 = np.array([1, 0, 0])
        want_p = np.array([q0, q1])
        c = Spring(k=1, L=1, q=lambda: [(q0, None), (q1, None)])

        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p)

    def test_p0_moved(self):
        q0 = np.array([0, 0, 0])
        q0_lst = [np.copy(q0)]
        q1 = np.array([1, 0, 0])
        c = Spring(k=1, L=1, q=lambda: [(q0_lst[0], None), (q1, None)])

        want_p = np.array([q0, q1])
        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p)
        # now move q0 in the list a bit.
        q0_new = np.array([10, 0, 0])
        q0_lst[0] = q0_new
        want_p_new = np.array([[6, 0, 0], [5, 0, 0]])
        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p_new)
        # again, move p0 in the list a bit.
        q0_new = np.array([-10, 0, 0])
        q0_lst[0] = q0_new
        want_p_new = np.array([[-5, 0, 0], [-4, 0, 0]])
        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p_new)

    def test_different_orientations(self):
        q0 = np.array([0, 0, 0])
        q1 = np.array([1.5, 1.5, 1.5])
        L = np.sqrt(3)
        want_p = np.array([0.75 - hlp.unit(np.ones(3)) * L/2,
                          0.75 + hlp.unit(np.ones(3)) * L/2])
        c = Spring(k=1, L=L, q=lambda: [(q0, None), (q1, None)])
        np.testing.assert_array_almost_equal(
            [p for (p, i) in c.project()], want_p)
        np.testing.assert_almost_equal(
            la.norm(c.project()[0][0] - c.project()[1][0]), L)


class TestParamters(unittest.TestCase):
    def test_parameters(self):
        q0 = np.array([0, 0, 0])
        q1 = np.array([1, 1, 1])
        k = 0.7
        L = 0.4

        c = Spring(k=k, L=L, q=lambda: [(q0, None), (q1, None)])
        np.testing.assert_almost_equal(c.w, k)
        np.testing.assert_almost_equal(c.L, L)
        np.testing.assert_array_almost_equal(
            c.A, np.identity(const.D))
        np.testing.assert_array_almost_equal(
            c.B, np.identity(const.D))

        w = 0.9
        A = np.identity(const.D) * 10
        B = np.identity(const.D) * 20
        c = Spring(k=k, L=L, q=lambda: [(q0, None), (q1, None)], w=w, A=A, B=B)
        np.testing.assert_almost_equal(c.w, k * w)
        np.testing.assert_almost_equal(c.L, L)
        np.testing.assert_array_almost_equal(c.A, A)
        np.testing.assert_array_almost_equal(c.B, B)


if __name__ == '__main__':
    unittest.main()
