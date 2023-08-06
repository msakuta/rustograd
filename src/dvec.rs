use std::collections::VecDeque;

#[derive(Clone, PartialEq, Debug)]
/// A dynamic dual number of arbitrary order, translated from C++ in a paper [^1]
///
/// [^1]: Higher Order Automatic Differentiation with Dual Numbers (doi: https://doi.org/10.3311/PPee.16341)
pub struct Dvec(VecDeque<f64>);

impl Dvec {
    pub fn new_n(v: f64, d: f64, n: usize) -> Self {
        let mut f = VecDeque::new();
        f.resize(n + 1, 0.);
        f[0] = v;
        if n >= 1 {
            f[1] = d
        };
        Self(f)
    }

    fn new(v: f64, mut d: Dvec) -> Self {
        d.0.push_front(v);
        d
    }

    pub fn is_real(&self) -> bool {
        self.0.len() == 1
    }

    fn F(&self) -> Dvec {
        // Front operator
        let mut ffront = self.clone();
        ffront.0.pop_back();
        ffront
    }

    fn D(&self) -> Dvec {
        // Derivation operator
        let mut fback = self.clone();
        fback.0.pop_front();
        fback
    }

    pub fn cos(&self) -> Self {
        if self.is_real() {
            single(self.0[0].cos())
        } else {
            Self::new(self.0[0].cos(), &-&(&self.F()).sin() * &self.D())
        }
    }

    pub fn sin(&self) -> Self {
        if self.is_real() {
            single(self.0[0].sin())
        } else {
            Self::new(self.0[0].sin(), &(&self.F()).cos() * &self.D())
        }
    }

    pub fn exp(&self) -> Self {
        if self.is_real() {
            single(self.0[0].exp())
        } else {
            Self::new(self.0[0].exp(), &(&self.F()).exp() * &self.D())
        }
    }
}

fn single(e: f64) -> Dvec {
    Dvec::new_n(e, 0., 0)
}

impl std::fmt::Display for Dvec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for v in &self.0 {
            write!(f, "{}, ", v)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl std::ops::Index<usize> for Dvec {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Add for &Dvec {
    type Output = Dvec;
    fn add(self, rhs: &Dvec) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0] + rhs.0[0])
        } else {
            Dvec::new(self.0[0] + rhs.0[0], &self.D() + &rhs.D())
        }
    }
}

impl std::ops::Sub for &Dvec {
    type Output = Dvec;
    fn sub(self, rhs: &Dvec) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0] - rhs.0[0])
        } else {
            Dvec::new(self.0[0] - rhs.0[0], &self.D() - &rhs.D())
        }
    }
}

impl std::ops::Mul for &Dvec {
    type Output = Dvec;
    fn mul(self, rhs: &Dvec) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0] * rhs.0[0])
        } else {
            Dvec::new(
                self.0[0] * rhs.0[0],
                &(&self.D() * &rhs.F()) + &(&self.F() * &rhs.D()),
            )
        }
    }
}

impl std::ops::Div for Dvec {
    type Output = Self;
    fn div(self, rhs: Dvec) -> Dvec {
        if self.is_real() || rhs.is_real() {
            single(self.0[0] / rhs.0[0])
        } else {
            Dvec::new(
                self.0[0] / rhs.0[0],
                (&(&self.D() * &rhs) - &(&self * &rhs.D())) / (&rhs * &rhs),
            )
        }
    }
}

impl std::ops::Neg for &Dvec {
    type Output = Dvec;
    fn neg(self) -> Self::Output {
        let mut ret = self.clone();
        ret.0.iter_mut().for_each(|v| *v = -*v);
        ret
    }
}

#[test]
fn test_dvec() {
    let d1 = Dvec::new_n(0., 1., 1);
    assert_eq!(d1.D(), Dvec::new_n(1., 0., 0));
    let d2 = Dvec::new_n(2., 0., 0);
    let d3 = &d1 + &d2;
    assert_eq!(d3, Dvec::new_n(2., 0., 0));
    assert_eq!(&d1 * &d2, Dvec::new_n(0., 0., 0));
    let d4 = Dvec::new_n(0.5, 0., 0);
    assert_eq!(d3 / d4, Dvec::new_n(4., 0., 0));
    let d5 = Dvec::new_n(2., 0., 1);
    assert_eq!(&d1 * &d5, Dvec::new_n(0., 2., 1));
    let d6 = d1.sin();
    assert_eq!(d6, Dvec::new_n(0., 1., 1));
    assert_eq!(d1.cos(), Dvec::new_n(1., 0., 1));
}
