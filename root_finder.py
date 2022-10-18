"""
Function approximation via adaptive sampling + using it for root finding.
"""
# Pauli Virtanen <pav@iki.fi>, 2014

from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import interpolate


class AdaptiveSampler(object):
    def __init__(self, a, b, num_points):
        self.a = float(a)
        self.b = float(b)
        self.num_points = int(num_points)
        self.values = None
        self.num_iter = 0
        self.fed = True
        self.next_x = self._get()
        self.max_score = np.inf

    @property
    def running(self):
        return self.num_iter < self.num_points

    def __iter__(self):
        return self

    def next(self):
        if not self.fed:
            raise RuntimeError("feed() not called")
        if not self.running:
            raise StopIteration()
        x = self.next_x
        self.fed = False
        return x

    __next__ = next

    def _get(self):
        if self.values is None:
            return self.a

        pts = np.hstack((self.values['x'][:,None], self.values['y']))

        if pts.shape[0] == 1:
            return self.b
        elif pts.shape[0] == 2:
            return (self.a + self.b) / 2.0

        pts[~np.isfinite(pts)] = 0

        pts = pts - pts.mean(axis=0)
        scales = pts.std(axis=0)
        scales[scales==0] = 1.0
        pts /= scales

        dx = np.diff(pts[:,0])

        # Euclidean path length parameterization
        dy = np.diff(pts, axis=0)
        ds = np.sqrt(np.sum(abs(dy)**2, axis=1))
        ds[ds==0] = 1.0

        # Compute score
        score = self._score(pts, ds, dy)
        j = np.argmax(score)
        self.max_score = score[j]

        # Choose the longer segment of the two, in case the scoring
        # was done on segment pairs
        if score.size == dx.size - 1:
            if dx[j+1]*ds[j+1] > dx[j]*ds[j]:
                j += 1

        # Bisect interval
        xp = (self.values['x'][j] + self.values['x'][j+1]) / 2

        return xp

    def _score(self, pts, ds, dy):
        # Estimate integral of curvature, over each segment, \int_L ||d^2y/ds^2|| ds
        dyds = dy/ds[:,None]
        dds = (dyds[1:] - dyds[:-1])
        dds = np.sqrt(np.sum(abs(dds)**2, axis=1))

        # Interval with maximum curvature*interval size
        dx = np.diff(pts[:,0])
        score = np.maximum(dx[1:], dx[:-1])**0.5 * dds * (ds[1:] + ds[:-1])

        return score

    def feed(self, y, x=None):
        self.num_iter += 1

        if x is None:
            x = self.next_x
        y = np.asarray(y).ravel()

        if self.values is None:
            self.values = np.zeros(0, dtype=[('x', float), ('y', y.dtype, (y.size,))])

        self.values.resize(self.values.size + 1)
        self.values['x'][-1] = x
        self.values['y'][-1] = y
        self.values.sort(order='x')

        self.next_x = self._get()
        self.fed = True

    @classmethod
    def sample_function(cls, func, a, b, num_points=20):
        sampler = cls(a, b, num_points)

        for x in sampler:
            sampler.feed(func(x))

        return sampler.values['x'], sampler.values['y']

    def insert(self, x, y):
        if x.ndim != 1 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x, y have invalid shape")

        self.values = np.zeros(x.shape[0], dtype=[('x', float), ('y', y.dtype, (y.shape[1],))])
        self.values['x'] = x
        self.values['y'] = y
        self.values.sort(order='x')

        self.num_iter = self.values.shape[0]
        self.next_x = self._get()


class _RootSampler(AdaptiveSampler):
    def _score(self, pts, ds, dy):
        assert pts.shape[1] == 2

        kscale = 2.0

        x = pts[:,0]
        y = self.values.copy()['y'][:,0]
        yscale = y.ptp()
        if yscale != 0:
            y = y / yscale

        dx = np.diff(x)
        dy = np.diff(y)

        # Obtain error estimates by considering quadratic
        # perturbations on each interval, resulting to change k -> k +/- k_typ in
        # the slope at either end of the interval

        k = dy/dx
        k_typ = kscale * max(1.0, np.median(abs(k)))

        cp = +k_typ / dx
        cm = -k_typ / dx

        xp = (x[1:] + x[:-1]) / 2 - k/(2*cp)
        xm = (x[1:] + x[:-1]) / 2 - k/(2*cm)

        xp = np.clip(xp, x[:-1], x[1:])
        xm = np.clip(xm, x[:-1], x[1:])

        yp = y[:-1] + k*(xp - x[:-1]) + cp*(xp - x[:-1])*(xp - x[1:])
        ym = y[:-1] + k*(xm - x[:-1]) + cm*(xm - x[:-1])*(xm - x[1:])

        score = abs(yp - ym)

        score[~np.isfinite(score)] = 0

        # Discard intervals that don't appear to be able to cross zero
        m = (np.sign(yp) == np.sign(ym))
        score[m] = 0

        if False:
            # Debug plot
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(x, y, '.-', hold=1)
            for j in range(dx.size):
                xx = np.linspace(x[j], x[j+1], 100)
                y1 = y[j] + k[j]*(xx - x[j]) + cp[j]*(xx - x[j])*(xx - x[j+1])
                y2 = y[j] + k[j]*(xx - x[j]) + cm[j]*(xx - x[j])*(xx - x[j+1])
                plt.fill_between(xx, y1, y2, hold=1, alpha=0.4)
                plt.plot(xp[j], yp[j], 'k.', hold=1)
                plt.plot(xm[j], ym[j], 'k.', hold=1)
            plt.grid(True)
            plt.axhline(0, c='k')
            plt.show()
            raw_input("pass")

        return score


def find_all_roots(func, a, b, tol=1e-6, points=()):
    sampler = _RootSampler(a, b, int(1e9))

    def iter_kinds():
        while True:
            yield 'sample'
            yield 1
            #yield 'term'
            #yield 'sample'
            yield 3
            yield 'term'

    kind_iter = iter_kinds()

    last_roots = None
    roots = np.array([], dtype=float)

    while True:
        kind = next(kind_iter)

        if sampler.num_iter < 4:
            x = None
            y = None
            n = 0
        else:
            xp = sampler.values['x']
            yp = sampler.values['y'].ravel()
            m = np.isfinite(yp)
            xp = xp[m]
            yp = yp[m]
            n = xp.size

        if kind == 'term':
            if n > 6:
                # Termination condition
                tck = interpolate.splrep(xp, yp, k=1)
                assert np.isfinite(tck[1]).all()
                pp = interpolate.PPoly.from_spline(tck, extrapolate=False)
                roots = pp.roots()

                if (last_roots is not None and
                    last_roots.size == roots.size and
                    np.all([abs(r - xp).min() < tol for r in roots]) and
                    sampler.max_score < tol**(1./4)):
                    break
            continue
        elif n < 6 or kind == 'sample':
            if n > 6 and points:
                # Feed in pre-given 'critical' points, if any
                for x in points:
                    if abs(x - xp).min() < tol:
                        continue
                    sampler.feed(func(x), x=x)
                points = ()

            if sampler.max_score > 0 or n < 3:
                x = sampler.next()
                sampler.feed(func(x), x=x)
            continue
        else:
            tck = interpolate.splrep(xp, yp, k=kind)
            assert np.isfinite(tck[1]).all()
            pp = interpolate.PPoly.from_spline(tck, extrapolate=False)
            last_roots = roots
            roots = pp.roots()
            for x in roots:
                if abs(x - xp).min() < tol:
                    continue
                sampler.feed(func(x), x=x)
            continue

    return roots, xp, yp


def demo_simple():
    def f(x):
        return x**2-1
    f = np.vectorize(f)

    x_root, xp, yp = find_all_roots(f, -10, 10)

    import matplotlib.pyplot as plt
    xx = np.linspace(-10, 10, 1000)
    plt.plot(xx, f(xx), '-', x_root, f(x_root), 'o', xp, yp, '.')
    plt.ylim(-5, 5)
    plt.legend(['f(x)', 'roots', 'points sampled ({0} total)'.format(len(xp))])
    plt.grid(True)
    plt.show()
    print(x_root[0])


if __name__ == "__main__":
    demo_simple()