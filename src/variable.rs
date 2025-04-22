use super::*;
use rand::Rng;
use rhai::{CustomType, TypeBuilder};
use std::sync::Arc;

#[derive(Clone)]
pub struct Variable {
    pub name: String,
    #[allow(dead_code)]
    prior: VariableExpr,
    conditionals: Arc<RwLock<Vec<Conditional>>>,
    posterior: VariableExpr,
}

impl Variable {
    pub fn new(name: String, prior: VariableExpr, conditionals: Vec<Conditional>) -> Self {
        let conditionals = Arc::new(RwLock::new(conditionals));
        let posterior = Self::make_posterior(prior.clone(), conditionals.clone());

        Variable {
            name,
            prior,
            conditionals,
            posterior,
        }
    }
    fn make_posterior(
        prior: VariableExpr,
        conditionals: Arc<RwLock<Vec<Conditional>>>,
    ) -> VariableExpr {
        VariableExpr::new(move |engine, generation, time| {
            let condition_lock = conditionals.read();
            let mut values = vec![];
            let mut total_weight = 0.0;
            for condition in condition_lock.iter() {
                if condition.condition.eval(engine, generation, time)? == 1.0 {
                    let weight = condition
                        .weight
                        .unwrap_or_else(|| 1.0 / condition_lock.len() as f64);
                    total_weight += weight;
                    values.push((weight, condition.value.eval(engine, generation, time)?));
                }
            }

            if total_weight == 0.0 {
                return prior.eval(engine, generation, time);
            }

            let selector: f64 = rand::rng().random_range(0.0..total_weight);
            let mut sum = 0.0;
            for w in values.iter() {
                sum += w.0;
                if sum >= selector {
                    return Ok(w.1);
                }
            }

            // shouldn't ever happen
            Ok(values.last().unwrap().1)
        })
    }

    pub fn add_conditional(&self, condition: Conditional) {
        self.conditionals.write().push(condition);
    }

    pub fn posterior(&self) -> VariableExpr {
        self.posterior.clone()
    }
}

#[derive(Clone)]
pub struct Conditional {
    pub weight: Option<f64>,
    pub condition: VariableExpr,
    pub value: VariableExpr,
}

#[derive(Clone)]
pub struct VariableExpr {
    pub key: u128,
    xform: Arc<dyn Fn(&EngineRef, usize, usize) -> Result<f64, String>>,
}

impl VariableExpr {
    pub fn new<F: Fn(&EngineRef, usize, usize) -> Result<f64, String> + 'static>(
        f: F,
    ) -> VariableExpr {
        VariableExpr {
            key: uuid::Uuid::now_v7().as_u128(),
            xform: Arc::new(move |engine: &EngineRef, generation: usize, time: usize| {
                f(engine, generation, time)
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
    pub fn atan2_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.atan2(other), self)
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
    pub fn copysign_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.copysign(other), self)
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

    pub fn div_euclid_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.div_euclid(other), self)
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

    pub fn hypot_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.hypot(other), self)
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
    pub fn log_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.log(other), self)
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
    pub fn max_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.max(other), self)
    }
    pub fn maximum(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.maximum(y), self, other)
    }
    pub fn maximum_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.maximum(other), self)
    }
    pub fn midpoint(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.midpoint(y), self, other)
    }
    pub fn midpoint_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.midpoint(other), self)
    }
    pub fn min(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.min(y), self, other)
    }
    pub fn min_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.min(other), self)
    }

    pub fn minimum(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.minimum(y), self, other)
    }

    pub fn minimum_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.minimum(other), self)
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
    pub fn powf_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.powf(other), self)
    }
    pub fn recip(self) -> VariableExpr {
        apply1(|x| x.recip(), self)
    }
    pub fn rem_euclid(self, other: VariableExpr) -> VariableExpr {
        apply2(|x, y| x.rem_euclid(y), self, other)
    }
    pub fn rem_euclid_f(self, other: f64) -> VariableExpr {
        apply1(move |x| x.rem_euclid(other), self)
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
    pub fn geq(self, other: VariableExpr) -> VariableExpr {
        apply2(move |x, y| if x >= y { 1.0 } else { 0.0 }, self, other)
    }
    pub fn geq_f(self, a: f64) -> VariableExpr {
        apply1(move |x| if x >= a { 1.0 } else { 0.0 }, self)
    }
    pub fn leq(self, other: VariableExpr) -> VariableExpr {
        apply2(move |x, y| if x <= y { 1.0 } else { 0.0 }, self, other)
    }
    pub fn leq_f(self, a: f64) -> VariableExpr {
        apply1(move |x| if x <= a { 1.0 } else { 0.0 }, self)
    }
    pub fn mul(self, other: VariableExpr) -> VariableExpr {
        apply2(move |x, y| x * y, self, other)
    }
    pub fn mul_f(self, a: f64) -> VariableExpr {
        apply1(move |x| x * a, self)
    }

    pub fn add(self, other: VariableExpr) -> VariableExpr {
        apply2(move |x, y| x + y, self, other)
    }
    pub fn add_f(self, a: f64) -> VariableExpr {
        apply1(move |x| x + a, self)
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
    pub fn eval(
        &self,
        engine: &EngineRef,
        generation: usize,
        timestep: usize,
    ) -> Result<f64, String> {
        self.posterior.eval(engine, generation, timestep)
    }

    pub fn eval_n(
        &self,
        engine: &EngineRef,
        timestep: usize,
        n: usize,
    ) -> Result<Vec<f64>, String> {
        self.posterior.eval_n(engine, timestep, n)
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
            .with_fn("constant", constant)
            .with_fn("uniform", uniform)
            .with_fn("uniform", uniform)
            .with_fn("uniform", uniform)
            .with_fn("normal", normal)
            .with_fn("bernoulli", bernoulli)
            .with_fn("poisson", poisson)
            .with_fn("exponential", exponential)
            // .with_fn("choice", choice)
            .with_fn("pareto", pareto)
            .with_fn("skew_normal", skew_normal)
            .with_fn("normal_range", normal_range)
            .with_fn("log_normal_range", log_normal_range)
            .with_fn("skewed_log_normal", skewed_log_normal)
            .with_fn(
                "log_normal_range_pct_deviation_at",
                log_normal_range_pct_deviation_at,
            )
            // Operators
            .with_fn("==", a2(|x, y| indicator(x == y)))
            .with_fn("!=", a2(|x, y| indicator(x != y)))
            .with_fn(">", a2(|x, y| indicator(x > y)))
            .with_fn(">=", a2(|x, y| indicator(x >= y)))
            .with_fn("<", a2(|x, y| indicator(x < y)))
            .with_fn("<=", a2(|x, y| indicator(x <= y)))
            .with_fn("!", a1(|x| 1.0 - x))
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
            // Normal functions
            .with_fn("abs", VariableExpr::abs)
            .with_fn("acos", VariableExpr::acos)
            .with_fn("acosh", VariableExpr::acosh)
            .with_fn("asin", VariableExpr::asin)
            .with_fn("atan", VariableExpr::atan)
            .with_fn("atan2", VariableExpr::atan2)
            .with_fn("atan2", VariableExpr::atan2_f)
            .with_fn("cbrt", VariableExpr::cbrt)
            .with_fn("ceil", VariableExpr::ceil)
            .with_fn("clamp", VariableExpr::clamp)
            .with_fn("copysign", VariableExpr::copysign)
            .with_fn("copysign", VariableExpr::copysign_f)
            .with_fn("cos", VariableExpr::cos)
            .with_fn("cosh", VariableExpr::cosh)
            .with_fn("div_euclid", VariableExpr::div_euclid)
            .with_fn("div_euclid", VariableExpr::div_euclid_f)
            .with_fn("exp", VariableExpr::exp)
            .with_fn("exp2", VariableExpr::exp2)
            .with_fn("exp_m1", VariableExpr::exp_m1)
            .with_fn("floor", VariableExpr::floor)
            .with_fn("hypot", VariableExpr::hypot)
            .with_fn("hypot", VariableExpr::hypot_f)
            .with_fn("is_finite", VariableExpr::is_finite)
            .with_fn("is_infinite", VariableExpr::is_infinite)
            .with_fn("is_nan", VariableExpr::is_nan)
            .with_fn("is_normal", VariableExpr::is_normal)
            .with_fn("is_sign_negative", VariableExpr::is_sign_negative)
            .with_fn("is_sign_positive", VariableExpr::is_sign_positive)
            .with_fn("is_subnormal", VariableExpr::is_subnormal)
            .with_fn("ln", VariableExpr::ln)
            .with_fn("ln_1p", VariableExpr::ln_1p)
            .with_fn("log", VariableExpr::log)
            .with_fn("log", VariableExpr::log_f)
            .with_fn("log10", VariableExpr::log10)
            .with_fn("log2", VariableExpr::log2)
            .with_fn("max", VariableExpr::max)
            .with_fn("max", VariableExpr::max_f)
            .with_fn("maximum", VariableExpr::maximum)
            .with_fn("maximum", VariableExpr::maximum_f)
            .with_fn("midpoint", VariableExpr::midpoint)
            .with_fn("midpoint", VariableExpr::midpoint_f)
            .with_fn("min", VariableExpr::min)
            .with_fn("min", VariableExpr::min_f)
            .with_fn("minimum", VariableExpr::minimum)
            .with_fn("minimum", VariableExpr::minimum_f)
            .with_fn("next_down", VariableExpr::next_down)
            .with_fn("next_up", VariableExpr::next_up)
            .with_fn("powf", VariableExpr::powf)
            .with_fn("powf", VariableExpr::powf_f)
            .with_fn("recip", VariableExpr::recip)
            .with_fn("rem_euclid", VariableExpr::rem_euclid)
            .with_fn("rem_euclid", VariableExpr::rem_euclid_f)
            .with_fn("round", VariableExpr::round)
            .with_fn("round_ties_even", VariableExpr::round_ties_even)
            .with_fn("signum", VariableExpr::signum)
            .with_fn("sin", VariableExpr::sin)
            .with_fn("sinh", VariableExpr::sinh)
            .with_fn("sqrt", VariableExpr::sqrt)
            .with_fn("tan", VariableExpr::tan)
            .with_fn("tanh", VariableExpr::tanh)
            .with_fn("to_degrees", VariableExpr::to_degrees)
            .with_fn("to_radians", VariableExpr::to_radians)
            .with_fn("geq", VariableExpr::geq)
            .with_fn("geq", VariableExpr::geq_f)
            .with_fn("leq", VariableExpr::leq)
            .with_fn("leq", VariableExpr::leq_f)
            .with_fn("mul", VariableExpr::mul)
            .with_fn("mul", VariableExpr::mul_f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let mut engine = Engine::new(None);
        let source = "\
        let u1 = uniform(0.0, 1.0);
        let u2 = uniform(0.1, 0.2);
        let u3 = u1 + u2;
        report(u3);
        ";

        engine.run(source).unwrap();
    }

    #[test]
    fn distributions() {
        let mut engine = Engine::new(None);
        let source = include_str!("../examples/distributions.rm");

        engine.run(source).unwrap();
    }

    #[test]
    fn basic_functions() {
        let mut engine = Engine::new(None);
        let source = include_str!("../examples/basic_functions.rm");

        engine.run(source).unwrap();
    }
}
