use std::collections::VecDeque;

use crate::Tensor;

#[derive(Clone, PartialEq, Debug)]
/// A dynamic dual number of arbitrary order, translated from C++ in a paper [^1]
///
/// [^1]: Higher Order Automatic Differentiation with Dual Numbers (doi: https://doi.org/10.3311/PPee.16341)
pub struct Dvec<T = f64>(VecDeque<T>);

impl<T: Tensor> Dvec<T> {
    pub fn new_n(v: T, d: T, n: usize) -> Self {
        let mut f = VecDeque::new();
        f.resize(n + 1, T::default());
        f[0] = v;
        if n >= 1 {
            f[1] = d
        };
        Self(f)
    }

    fn new(v: T, mut d: Self) -> Self {
        d.0.push_front(v);
        d
    }

    pub fn is_real(&self) -> bool {
        self.0.len() == 1
    }

    fn f(&self) -> Self {
        // Front operator
        let mut ffront = self.clone();
        ffront.0.pop_back();
        ffront
    }

    fn d(&self) -> Self {
        // Derivation operator
        let mut fback = self.clone();
        fback.0.pop_front();
        fback
    }

    /// Apply a function with arbitrary number of derivatives.
    /// You need to have a function that can be derived infinite times.
    /// The second argument to the function is the order of differentiation
    ///
    /// # Example
    ///
    /// ```
    /// d1.apply(|x, n| {
    ///     match n % 4 {
    ///         0 => x.sin(),
    ///         1 => x.cos(),
    ///         2 => -x.sin(),
    ///         3 => -x.cos(),
    ///         _ => unreachable!(),
    ///     }
    /// });
    /// ```
    pub fn apply(&self, f: fn(T, usize) -> T) -> Self {
        self.apply_rec(f, 0)
    }

    fn apply_rec(&self, f: fn(T, usize) -> T, n: usize) -> Self {
        if self.is_real() {
            single(f(self.0[0].clone(), n))
        } else {
            Self::new(
                f(self.0[0].clone(), n),
                &self.f().apply_rec(f, n + 1) * &self.d(),
            )
        }
    }
}

impl Dvec<f64> {
    pub fn cos(&self) -> Self {
        if self.is_real() {
            single(self.0[0].cos())
        } else {
            Self::new(self.0[0].cos(), &-&(&self.f()).sin() * &self.d())
        }
    }

    pub fn sin(&self) -> Self {
        if self.is_real() {
            single(self.0[0].sin())
        } else {
            Self::new(self.0[0].sin(), &(&self.f()).cos() * &self.d())
        }
    }

    pub fn exp(&self) -> Self {
        if self.is_real() {
            single(self.0[0].exp())
        } else {
            Self::new(self.0[0].exp(), &(&self.f()).exp() * &self.d())
        }
    }
}

fn single<T: Tensor>(e: T) -> Dvec<T> {
    Dvec::new_n(e, T::default(), 0)
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

impl<T: Tensor> std::ops::Index<usize> for Dvec<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Tensor> std::ops::Add for &Dvec<T> {
    type Output = Dvec<T>;
    fn add(self, rhs: &Dvec<T>) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0].clone() + rhs.0[0].clone())
        } else {
            Dvec::new(self.0[0].clone() + rhs.0[0].clone(), &self.d() + &rhs.d())
        }
    }
}

impl<T: Tensor> std::ops::Sub for &Dvec<T> {
    type Output = Dvec<T>;
    fn sub(self, rhs: &Dvec<T>) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0].clone() - rhs.0[0].clone())
        } else {
            Dvec::new(self.0[0].clone() - rhs.0[0].clone(), &self.d() - &rhs.d())
        }
    }
}

impl<T: Tensor> std::ops::Mul for &Dvec<T> {
    type Output = Dvec<T>;
    fn mul(self, rhs: &Dvec<T>) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0].clone() * rhs.0[0].clone())
        } else {
            Dvec::new(
                self.0[0].clone() * rhs.0[0].clone(),
                &(&self.d() * &rhs.f()) + &(&self.f() * &rhs.d()),
            )
        }
    }
}

impl<T: Tensor> std::ops::Div for Dvec<T> {
    type Output = Dvec<T>;
    fn div(self, rhs: Dvec<T>) -> Self::Output {
        if self.is_real() || rhs.is_real() {
            single(self.0[0].clone() / rhs.0[0].clone())
        } else {
            Dvec::new(
                self.0[0].clone() / rhs.0[0].clone(),
                (&(&self.d() * &rhs) - &(&self * &rhs.d())) / (&rhs * &rhs),
            )
        }
    }
}

impl<T: Tensor> std::ops::Neg for &Dvec<T> {
    type Output = Dvec<T>;
    fn neg(self) -> Self::Output {
        let mut ret = self.clone();
        ret.0.iter_mut().for_each(|v| *v = -std::mem::take(v));
        ret
    }
}

#[test]
fn test_dvec() {
    let d1 = Dvec::new_n(0., 1., 1);
    assert_eq!(d1.d(), Dvec::new_n(1., 0., 0));
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
