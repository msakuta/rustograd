use crate::dnum::choose;

#[derive(Clone, PartialEq, Debug)]
/// A 2-variate extension to [`crate::Dnum`], translated from C++ in a paper [^1]
///
/// [^1]: Higher Order Automatic Differentiation with Dual Numbers (doi: https://doi.org/10.3311/PPee.16341)
pub struct Dnum2<const N: usize> {
    f: Vec<f64>, // Dynamic array instead of fixed, since Rust can't have const expression in array size
}

impl<const N: usize> Dnum2<N> {
    const N2: usize = (N + 1) * (N + 2) / 2;
    pub fn new(v: f64, du: f64, dv: f64) -> Self {
        assert!(
            2 <= N,
            "This constructor can only be used with 2 or more dual numbers"
        );
        let mut f = vec![0.; Self::N2]; // triangle matrix
        f[0] = v;
        f[1] = du;
        f[N + 1] = dv;
        Self { f }
    }

    pub fn is_real(&self) -> bool {
        self.f.iter().skip(1).all(|&j| j == 0.)
    }

    pub fn conjugate(&self) -> Self {
        let res = self
            .f
            .iter()
            .enumerate()
            .map(|(i, &v)| if i == 0 { v } else { -v })
            .collect();
        Self { f: res }
    }

    fn index_of(j: usize, l: usize) -> usize {
        let ij = j as isize;
        (ij * (N + 1) as isize - ij * (ij - 1) / 2) as usize + l
    }
}

impl<const N: usize> std::ops::Add for Dnum2<N> {
    type Output = Self;
    fn add(mut self, rhs: Dnum2<N>) -> Self::Output {
        for (selfv, rhsv) in self.f.iter_mut().zip(rhs.f.into_iter()) {
            *selfv = *selfv + rhsv;
        }
        self
    }
}

impl<const N: usize> std::ops::Div<f64> for Dnum2<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        let mut f = self.f;
        f.iter_mut().for_each(|j| *j = *j / rhs);
        Self { f }
    }
}

impl<const N: usize> std::ops::Index<(usize, usize)> for Dnum2<N> {
    type Output = f64;
    fn index(&self, (j, l): (usize, usize)) -> &Self::Output {
        &self.f[Self::index_of(j, l)]
    }
}

impl<const N: usize> std::ops::IndexMut<(usize, usize)> for Dnum2<N> {
    fn index_mut(&mut self, (j, l): (usize, usize)) -> &mut Self::Output {
        &mut self.f[Self::index_of(j, l)]
    }
}

impl<const N: usize> std::ops::Mul for Dnum2<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut f = Self {
            f: vec![0.; Self::N2],
        };
        for j in 0..=N {
            for l in 0..=N - j {
                for k in 0..=N {
                    for m in 0..=N - k {
                        if j + k + l + m < N {
                            // println!(
                            //     "[{dest}] <= ({j}, {k}) = {lhs}, ({l}, {m}) = {rhs}, {jk}, {lm}",
                            //     dest = Self::index_of(j, l),
                            //     lhs = self[(j, l)],
                            //     rhs = rhs[(k, m)],
                            //     jk = choose(j + k, j),
                            //     lm = choose(l + m, l)
                            // );
                            f[(j + k, l + m)] += self[(j, l)]
                                * rhs[(k, m)]
                                * choose(j + k, j) as f64
                                * choose(l + m, l) as f64;
                        }
                    }
                }
            }
        }
        f
    }
}

impl<const N: usize> std::ops::Div for Dnum2<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_real() {
            self / rhs.f[0]
        } else {
            let crhs = rhs.conjugate();
            let denom = rhs * crhs.clone();
            (self * crhs) / denom
        }
    }
}

#[test]
fn test_dual() {
    let d1 = Dnum2::<2>::new(1., 2., 3.);
    let d2 = Dnum2::<2>::new(3., 4., 5.);
    assert_eq!(d1 + d2, Dnum2::<2>::new(4., 6., 8.));

    let d3 = Dnum2::<3> {
        f: vec![1., 2., 3., 4., 5., 6.],
    };
    assert_eq!(
        d3.conjugate(),
        Dnum2::<3> {
            f: vec![1., -2., -3., -4., -5., -6.]
        }
    );
    assert_eq!(
        d3 / 2.,
        Dnum2::<3> {
            f: vec![0.5, 1., 1.5, 2., 2.5, 3.0]
        }
    );

    let d4 = Dnum2::<2>::new(1., 2., 3.);
    let d5 = Dnum2::<2>::new(30., 20., 10.);
    assert_eq!(d4.clone() * d5.clone(), Dnum2::<2>::new(30., 80., 100.));
    assert_eq!(d4.clone() * d5.clone(), d5.clone() * d4.clone());
    assert!(!d4.is_real());
    assert!(!d5.is_real());
    assert_eq!(d5 / d4, Dnum2::<2>::new(30., -40., -80.));
}
