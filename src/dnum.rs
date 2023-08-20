#[derive(Clone, Copy, PartialEq, Debug)]
/// An implementation of order N dual nuumber, translated from C++ in a paper [^1]
///
/// [^1]: Higher Order Automatic Differentiation with Dual Numbers (doi: https://doi.org/10.3311/PPee.16341)
pub struct Dnum<const N: usize> {
    f: [f64; N], // value and derivatives
}

impl<const N: usize> Dnum<N> {
    pub fn new(v: f64, d: f64) -> Self {
        assert!(
            2 <= N,
            "This constructor can only be used with 2 or more dual numbers"
        );
        let mut f = [0.; N];
        f[0] = v;
        f[1] = d;
        Self { f }
    }

    pub fn from_array(f: [f64; N]) -> Self {
        Self { f }
    }

    pub fn is_real(&self) -> bool {
        self.f.iter().skip(1).all(|&j| j == 0.)
    }

    pub fn conjugate(&self) -> Self {
        let mut res = [0.; N];
        for (resv, selfv) in res.iter_mut().zip(self.f.iter()).skip(1) {
            *resv = -selfv;
        }
        res[0] = self.f[0];
        Self { f: res }
    }

    pub fn exp(&self) -> Self {
        let mut result = *self;
        let ex = self.f[0].exp();
        result.f[0] = ex;
        for (i, v) in result.f.iter_mut().enumerate().skip(1) {
            *v *= ex.powi(i as i32);
        }
        result
    }
}

impl<const N: usize> std::ops::Add for Dnum<N> {
    type Output = Self;
    fn add(self, rhs: Dnum<N>) -> Self::Output {
        let mut f = [0.; N];
        for ((fv, selfv), rhsv) in f.iter_mut().zip(self.f.into_iter()).zip(rhs.f.into_iter()) {
            *fv = selfv + rhsv;
        }
        Self { f }
    }
}

impl<const N: usize> std::ops::Div<f64> for Dnum<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        let mut f = self.f;
        f.iter_mut().for_each(|j| *j = *j / rhs);
        Self { f }
    }
}

impl<const N: usize> std::ops::Mul for Dnum<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut f = [0.; N];
        for j in 0..N {
            for k in 0..N {
                if j + k < N {
                    f[j + k] += self.f[j] * rhs.f[k] * choose(j + k, j) as f64;
                }
            }
        }
        Self { f }
    }
}

impl<const N: usize> std::ops::Div for Dnum<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_real() {
            self / rhs.f[0]
        } else {
            let crhs = rhs.conjugate();
            (self * crhs) / (rhs * crhs)
        }
    }
}

impl<const N: usize> std::ops::Neg for Dnum<N> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        for v in self.f.iter_mut() {
            *v = -*v;
        }
        self
    }
}

impl<const N: usize> std::ops::Index<usize> for Dnum<N> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.f[index]
    }
}

#[test]
fn test_dual() {
    let d1 = Dnum::<2>::new(1., 2.);
    let d2 = Dnum::<2>::new(3., 4.);
    assert_eq!(d1 + d2, Dnum::<2>::new(4., 6.));

    let d3 = Dnum::<3> { f: [1., 2., 3.] };
    assert_eq!(d3.conjugate(), Dnum::<3> { f: [1., -2., -3.] });
    assert_eq!(d3 / 2., Dnum::<3> { f: [0.5, 1., 1.5] });

    let d4 = Dnum::<2>::new(1., 2.);
    let d5 = Dnum::<2>::new(20., -10.);
    assert_eq!(d4 * d5, Dnum::<2>::new(20., 30.));
    assert_eq!(d4 * d5, d5 * d4);
    assert!(!d4.is_real());
    assert!(!d5.is_real());
    assert_eq!(d5 / d4, Dnum::<2>::new(20., -50.));
}

#[test]
fn test_dual3() {
    let d1 = Dnum::<3>::new(1., 2.);
    let d2 = Dnum::<3>::new(20., -10.);
    assert_eq!(d2 / d1, Dnum::<3>::from_array([20., -50., 200.]));
}

pub(crate) fn choose(n: usize, k: usize) -> usize {
    assert!(k <= n);
    let mut res = 1;
    for i in 0..k {
        res *= n - i
    }
    for i in 2..=k {
        res /= i;
    }
    res
}

#[test]
fn test_choose() {
    assert_eq!(choose(0, 0), 1);
    assert_eq!(choose(0, 0), 1);
    assert_eq!(choose(1, 1), 1);
    assert_eq!(choose(2, 1), 2);
    assert_eq!(choose(2, 2), 1);
    assert_eq!(choose(3, 0), 1);
    assert_eq!(choose(3, 1), 3);
    assert_eq!(choose(3, 2), 3);
    assert_eq!(choose(3, 3), 1);
}
