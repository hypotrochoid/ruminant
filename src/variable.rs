use super::*;
use rhai::{CustomType, TypeBuilder};
use std::sync::Arc;

#[derive(Clone)]
pub struct Variable {
    pub name: String,
    exec: VariableExpr,
}

#[derive(Clone)]
pub struct VariableExpr {
    pub key: u128,
    xform: Arc<dyn Fn(&EngineRef, usize, usize) -> Result<f64, String>>,
}

impl VariableExpr {
    pub fn new<F: Fn(&EngineRef, usize, usize) -> f64 + 'static>(f: F) -> VariableExpr {
        VariableExpr {
            key: uuid::Uuid::now_v7().as_u128(),
            xform: Arc::new(move |engine: &EngineRef, generation: usize, time: usize| {
                Ok(f(engine, generation, time))
            }),
        }
    }

    pub fn eval(
        &self,
        engine: &EngineRef,
        generation: usize,
        timestep: usize,
    ) -> Result<f64, String> {
        let key = ParticleKey {
            generation,
            timestep,
            variable: self.key,
        };

        {
            let lock = engine.engine.particles.read();

            if let Some(p) = lock.get(&key) {
                return Ok(p.value);
            }
        }

        let value = (*self.xform)(engine, generation, timestep)?;

        {
            let mut lock = engine.engine.particles.write();
            lock.insert(key, Particle { value });
        }

        Ok(value)
    }
    pub fn eval_n(
        &self,
        engine: &EngineRef,
        timestep: usize,
        n: usize,
    ) -> Result<Vec<f64>, String> {
        (0..n).map(|i| self.eval(engine, i, timestep)).collect()
    }
}

pub fn apply1<F: Fn(f64) -> f64 + 'static>(f: F, v: VariableExpr) -> VariableExpr {
    VariableExpr {
        key: uuid::Uuid::now_v7().as_u128(),
        xform: Arc::new(move |engine: &EngineRef, generation: usize, time: usize| {
            Ok(f((*v.xform)(engine, generation, time)?))
        }),
    }
}

pub fn apply2<F: Fn(f64, f64) -> f64 + 'static>(
    f: F,
    v1: VariableExpr,
    v2: VariableExpr,
) -> VariableExpr {
    VariableExpr {
        key: uuid::Uuid::now_v7().as_u128(),
        xform: Arc::new(move |engine: &EngineRef, generation: usize, time: usize| {
            Ok(f(
                (*v1.xform)(engine, generation, time)?,
                (*v2.xform)(engine, generation, time)?,
            ))
        }),
    }
}

impl VariableExpr {
    pub fn abs(self) -> VariableExpr {
        apply1(|x| x.abs(), self)
    }
    pub fn acos(self) -> VariableExpr {
        apply1(|x| x.acos(), self)
    }
    pub fn acosh(self) -> VariableExpr {
        apply1(|x| x.acosh(), self)
    }
    pub fn asin(self) -> VariableExpr {
        apply1(|x| x.asin(), self)
    }
    pub fn asinh(self) -> VariableExpr {
        apply1(|x| x.asinh(), self)
    }
    pub fn atan(self) -> VariableExpr {
        apply1(|x| x.atan(), self)
    }
    pub fn atan2(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.atan2(y), self, other)
    }
    pub fn atanh(self) -> VariableExpr {
        apply1(|x| x.atanh(), self)
    }
    pub fn cbrt(self) -> VariableExpr {
        apply1(|x| x.cbrt(), self)
    }
    pub fn ceil(self) -> VariableExpr {
        apply1(|x| x.ceil(), self)
    }
    pub fn clamp(self, a: f64, b: f64) -> VariableExpr {
        apply1(move |x| x.clamp(a, b), self)
    }
    pub fn copysign(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.copysign(y), self, other)
    }
    pub fn cos(self) -> VariableExpr {
        apply1(|x| x.cos(), self)
    }
    pub fn cosh(self) -> VariableExpr {
        apply1(|x| x.cosh(), self)
    }
    pub fn div_euclid(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.div_euclid(y), self, other)
    }
    pub fn exp(self) -> VariableExpr {
        apply1(|x| x.exp(), self)
    }
    pub fn exp2(self) -> VariableExpr {
        apply1(|x| x.exp2(), self)
    }
    pub fn exp_m1(self) -> VariableExpr {
        apply1(|x| x.exp_m1(), self)
    }
    pub fn floor(self) -> VariableExpr {
        apply1(|x| x.floor(), self)
    }
    pub fn fract(self) -> VariableExpr {
        apply1(|x| x.fract(), self)
    }
    pub fn hypot(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.hypot(y), self, other)
    }

    pub fn is_finite(self) -> VariableExpr {
        apply1(|x| fb(x.is_finite()), self)
    }
    pub fn is_infinite(self) -> VariableExpr {
        apply1(|x| fb(x.is_infinite()), self)
    }
    pub fn is_nan(self) -> VariableExpr {
        apply1(|x| fb(x.is_nan()), self)
    }
    pub fn is_normal(self) -> VariableExpr {
        apply1(|x| fb(x.is_normal()), self)
    }
    pub fn is_sign_negative(self) -> VariableExpr {
        apply1(|x| fb(x.is_sign_negative()), self)
    }
    pub fn is_sign_positive(self) -> VariableExpr {
        apply1(|x| fb(x.is_sign_positive()), self)
    }
    pub fn is_subnormal(self) -> VariableExpr {
        apply1(|x| fb(x.is_subnormal()), self)
    }

    pub fn ln(self) -> VariableExpr {
        apply1(|x| x.ln(), self)
    }
    pub fn ln_1p(self) -> VariableExpr {
        apply1(|x| x.ln_1p(), self)
    }
    pub fn log(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.log(y), self, other)
    }
    pub fn log10(self) -> VariableExpr {
        apply1(|x| x.log10(), self)
    }
    pub fn log2(self) -> VariableExpr {
        apply1(|x| x.log2(), self)
    }
    pub fn max(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.max(y), self, other)
    }
    pub fn maximum(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.maximum(y), self, other)
    }
    pub fn midpoint(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.midpoint(y), self, other)
    }
    pub fn min(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.min(y), self, other)
    }
    pub fn minimum(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.minimum(y), self, other)
    }
    pub fn next_down(self) -> VariableExpr {
        apply1(|x| x.next_down(), self)
    }
    pub fn next_up(self) -> VariableExpr {
        apply1(|x| x.next_up(), self)
    }
    pub fn powf(self, other: Self) -> VariableExpr {
        apply2(|x, y| x.powf(y), self, other)
    }
    pub fn recip(self) -> VariableExpr {
        apply1(|x| x.recip(), self)
    }
    pub fn rem_euclid(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.rem_euclid(y), self, other)
    }
    pub fn round(self) -> VariableExpr {
        apply1(|x| x.round(), self)
    }
    pub fn round_ties_even(self) -> VariableExpr {
        apply1(|x| x.round_ties_even(), self)
    }
    pub fn signum(self) -> VariableExpr {
        apply1(|x| x.signum(), self)
    }
    pub fn sin(self) -> VariableExpr {
        apply1(|x| x.sin(), self)
    }
    pub fn sinh(self) -> VariableExpr {
        apply1(|x| x.sinh(), self)
    }
    pub fn sqrt(self) -> VariableExpr {
        apply1(|x| x.sqrt(), self)
    }
    pub fn tan(self) -> VariableExpr {
        apply1(|x| x.tan(), self)
    }
    pub fn tanh(self) -> VariableExpr {
        apply1(|x| x.tanh(), self)
    }
    pub fn to_degrees(self) -> VariableExpr {
        apply1(|x| x.to_degrees(), self)
    }
    pub fn to_radians(self) -> VariableExpr {
        apply1(|x| x.to_radians(), self)
    }
    pub fn geq(self, a: f64) -> VariableExpr {
        apply1(move |x| if x >= a { 1.0 } else { 0.0 }, self)
    }
    pub fn leq(self, a: f64) -> VariableExpr {
        apply1(move |x| if x <= a { 1.0 } else { 0.0 }, self)
    }
    pub fn mul(self, a: f64) -> VariableExpr {
        apply1(move |x| x * a, self)
    }
}

pub fn a1<F: Fn(f64) -> f64 + Clone + 'static>(f: F) -> impl Fn(VariableExpr) -> VariableExpr {
    move |v| apply1(f.clone(), v)
}
pub fn a2<F: Fn(f64, f64) -> f64 + Clone + 'static>(
    f: F,
) -> impl Fn(VariableExpr, VariableExpr) -> VariableExpr {
    move |v1, v2| apply2(f.clone(), v1.clone(), v2.clone())
}

pub fn indicator(x: bool) -> f64 {
    if x { 1.0 } else { 0.0 }
}

pub fn deindicator(x: f64) -> bool {
    x != 0.0
}

impl Variable {
    pub fn new(name: String, expr: VariableExpr) -> Variable {
        Variable { name, exec: expr }
    }

    pub fn eval(
        &self,
        engine: &EngineRef,
        generation: usize,
        timestep: usize,
    ) -> Result<f64, String> {
        self.exec.eval(engine, generation, timestep)
    }

    pub fn eval_n(
        &self,
        engine: &EngineRef,
        timestep: usize,
        n: usize,
    ) -> Result<Vec<f64>, String> {
        self.exec.eval_n(engine, timestep, n)
    }
    // op ==(int, int) -> bool;
    // op !=(int, int) -> bool;
    // op >(int, int) -> bool;
    // op >=(int, int) -> bool;
    // op <(int, int) -> bool;
    // op <=(int, int) -> bool;
    // op &(int, int) -> int;
    // op |(int, int) -> int;
    // op ^(int, int) -> int;
    // op ..(int, int) -> Range<int>;
    // op ..=(int, int) -> RangeInclusive<int>;
    // op +(int, int) -> int;
    // op -(int, int) -> int;
    // op *(int, int) -> int;
    // op /(int, int) -> int;
    // op %(int, int) -> int;
    // op **(int, int) -> int;
    // op >>(int, int) -> int;
    // op <<(int, int) -> int;
}

impl CustomType for VariableExpr {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("RandomVariable")
            // Constructors
            .with_fn("uniform", uniform)
            // Operators
            .with_fn("==", a2(|x, y| indicator(x == y)))
            .with_fn("!=", a2(|x, y| indicator(x != y)))
            .with_fn(">", a2(|x, y| indicator(x > y)))
            .with_fn(">=", a2(|x, y| indicator(x >= y)))
            .with_fn("<", a2(|x, y| indicator(x < y)))
            .with_fn("<=", a2(|x, y| indicator(x <= y)))
            .with_fn("&", a2(|x, y| indicator(deindicator(x) & deindicator(y))))
            .with_fn("|", a2(|x, y| indicator(deindicator(x) | deindicator(y))))
            .with_fn("^", a2(|x, y| x.powf(y)))
            .with_fn("..", |x, y| uniform(x, y))
            .with_fn("..=", |x, y| uniform(x, y))
            .with_fn("+", a2(|x, y| x + y))
            .with_fn("-", a2(|x, y| x - y))
            .with_fn("*", a2(|x, y| x * y))
            .with_fn("/", a2(|x, y| x / y))
            .with_fn("%", a2(|x, y| x % y))
            .with_fn("**", a2(|x, y| x.powf(y)))
            // .with_fn(">>", a2(|x, y| x >> y))
            // .with_fn("<<", a2(|x, y| x << y))
        ;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let mut engine = Engine::new();
        let source = "\
        let u1 = uniform(0.0, 1.0);
        let u2 = uniform(0.1, 0.2);
        let u3 = u1 + u2;
        report(u3);
        ";

        engine.run(source).unwrap();
    }
}
