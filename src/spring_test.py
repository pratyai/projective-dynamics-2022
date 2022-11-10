import unittest
import spring
import numpy as np
import numpy.linalg as la
import constraint as con


class TestProjection(unittest.TestCase):
    def test_compressed(self):
        p0 = np.array([0, 0, 0])
        q = np.array([0.5, 0, 0])
        want_p = np.array([1, 0, 0])
        c = spring.Spring(k=1, L=1, p0=lambda: (p0, None))

        np.testing.assert_array_almost_equal(
            c.project(q), want_p)

    def test_expanded(self):
        p0 = np.array([0, 0, 0])
        q = np.array([1.5, 0, 0])
        want_p = np.array([1, 0, 0])
        c = spring.Spring(k=1, L=1, p0=lambda: (p0, None))

        np.testing.assert_array_almost_equal(
            c.project(q), want_p)

    def test_exact(self):
        p0 = np.array([0, 0, 0])
        q = np.array([1, 0, 0])
        want_p = np.array([1, 0, 0])
        c = spring.Spring(k=1, L=1, p0=lambda: (p0, None))

        np.testing.assert_array_almost_equal(
            c.project(q), want_p)

    def test_p0_moved(self):
        p0 = np.array([0, 0, 0])
        p0_lst = [np.copy(p0)]
        q = np.array([1, 0, 0])
        c = spring.Spring(k=1, L=1, p0=lambda: (p0_lst[0], 0))

        want_p = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(
            c.project(q), want_p)
        # now move p0 in the list a bit.
        p0_new = np.array([10, 0, 0])
        p0_lst[0] = p0_new
        want_p_new = np.array([9, 0, 0])
        np.testing.assert_array_almost_equal(
            c.project(q), want_p_new)
        # again, move p0 in the list a bit.
        p0_new = np.array([-10, 0, 0])
        p0_lst[0] = p0_new
        want_p_new = np.array([-9, 0, 0])
        np.testing.assert_array_almost_equal(
            c.project(q), want_p_new)

    def test_different_orientations(self):
        p0 = np.array([0, 0, 0])
        q = np.array([1.5, 1.5, 1.5])
        want_p = np.array([1, 1, 1])
        L = np.sqrt(3)
        c = spring.Spring(k=1, L=L, p0=lambda: (p0, None))
        np.testing.assert_array_almost_equal(
            c.project(q), want_p)
        np.testing.assert_almost_equal(
            la.norm(c.project(np.array([1.5, 1.5, 1.5]))), L)
        np.testing.assert_almost_equal(
            la.norm(c.project(np.array([1.5, 2.5, 1.5]))), L)
        np.testing.assert_almost_equal(
            la.norm(c.project(np.array([2.5, 2.5, 1.5]))), L)
        np.testing.assert_almost_equal(
            la.norm(c.project(np.array([1.5, 0, 0]))), L)


class TestParamters(unittest.TestCase):
    def test_parameters(self):
        p0 = np.array([0, 0, 0])
        k = 0.7
        L = 0.4

        c = spring.Spring(k=k, L=L, p0=lambda: (p0, None))
        np.testing.assert_almost_equal(c.w, k)
        np.testing.assert_almost_equal(c.L, L)
        np.testing.assert_array_almost_equal(
            c.A, np.identity(con.Constraint.D))
        np.testing.assert_array_almost_equal(
            c.B, np.identity(con.Constraint.D))

        w = 0.9
        A = np.identity(con.Constraint.D) * 10
        B = np.identity(con.Constraint.D) * 20
        c = spring.Spring(k=k, L=L, p0=lambda: (p0, None), w=w, A=A, B=B)
        np.testing.assert_almost_equal(c.w, k * w)
        np.testing.assert_almost_equal(c.L, L)
        np.testing.assert_array_almost_equal(c.A, A)
        np.testing.assert_array_almost_equal(c.B, B)


if __name__ == '__main__':
    unittest.main()
