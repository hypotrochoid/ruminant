use {
    crate::{EmpiricalDist, VariableExpr, fb},
    itertools::Itertools,
    rhai::{Array, CustomType, Dynamic, TypeBuilder},
    std::{fs::File, io::read_to_string},
};

#[derive(Clone, Default, Debug)]
pub struct Sequence {
    values: Vec<f64>,
}

pub fn apply1_seq<F: Fn(f64) -> f64 + 'static>(f: F, v: Sequence) -> Sequence {
    Sequence {
        values: v.values.into_iter().map(f).collect(),
    }
}

pub fn apply2_seq<F: Fn(f64, f64) -> f64 + 'static>(f: F, v: Sequence, w: Sequence) -> Sequence {
    Sequence {
        values: v
            .values
            .into_iter()
            .zip(w.values.into_iter())
            .map(|(a, b)| f(a, b))
            .collect(),
    }
}

impl Sequence {
    fn dynamic_to_sample(value: Dynamic) -> Result<f64, String> {
        match value.type_name() {
            "i64" => Ok(value.cast::<i64>() as f64),
            "f64" => Ok(value.cast::<f64>()),
            "decimal" => Ok(value.cast::<f64>()),
            x => Err(format!("type {} cannot be used as a weight", x)),
        }
    }
    pub fn new(values: Array) -> Result<Self, String> {
        let v: Result<Vec<f64>, _> = values.into_iter().map(Self::dynamic_to_sample).collect();

        Ok(Sequence { values: v? })
    }

    pub fn diff(self) -> Sequence {
        let values: Vec<f64> = self
            .values
            .into_iter()
            .tuple_windows()
            .map(|(v1, v2)| v2 - v1)
            .collect();

        Sequence { values }
    }

    pub fn distribution(self) -> VariableExpr {
        let mut dist = EmpiricalDist::default();
        for val in self.values {
            dist.insert(val);
        }

        dist.distribution()
    }

    pub fn try_load(fname: &str) -> Result<Self, String> {
        let file = File::open(fname).map_err(|e| format!("{:?}", e))?;
        let values: Result<Vec<f64>, _> = read_to_string(file)
            .map_err(|e| format!("{:?}", e))?
            .lines()
            .map(str::parse::<f64>)
            .collect();

        Ok(Sequence {
            values: values.map_err(|e| format!("{:?}", e))?,
        })
    }

    pub fn load(fname: &str) -> Self {
        Sequence::try_load(fname).unwrap()
    }
    pub fn abs(self) -> Sequence {
        apply1_seq(|x| x.abs(), self)
    }
    pub fn acos(self) -> Sequence {
        apply1_seq(|x| x.acos(), self)
    }
    pub fn acosh(self) -> Sequence {
        apply1_seq(|x| x.acosh(), self)
    }
    pub fn asin(self) -> Sequence {
        apply1_seq(|x| x.asin(), self)
    }
    pub fn asinh(self) -> Sequence {
        apply1_seq(|x| x.asinh(), self)
    }
    pub fn atan(self) -> Sequence {
        apply1_seq(|x| x.atan(), self)
    }
    pub fn atan2(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.atan2(y), self, other)
    }
    pub fn atan2_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.atan2(other), self)
    }
    pub fn atanh(self) -> Sequence {
        apply1_seq(|x| x.atanh(), self)
    }
    pub fn cbrt(self) -> Sequence {
        apply1_seq(|x| x.cbrt(), self)
    }
    pub fn ceil(self) -> Sequence {
        apply1_seq(|x| x.ceil(), self)
    }
    pub fn clamp(self, a: f64, b: f64) -> Sequence {
        apply1_seq(move |x| x.clamp(a, b), self)
    }
    pub fn copysign(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.copysign(y), self, other)
    }
    pub fn copysign_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.copysign(other), self)
    }
    pub fn cos(self) -> Sequence {
        apply1_seq(|x| x.cos(), self)
    }
    pub fn cosh(self) -> Sequence {
        apply1_seq(|x| x.cosh(), self)
    }
    pub fn div_euclid(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.div_euclid(y), self, other)
    }

    pub fn div_euclid_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.div_euclid(other), self)
    }

    pub fn exp(self) -> Sequence {
        apply1_seq(|x| x.exp(), self)
    }
    pub fn exp2(self) -> Sequence {
        apply1_seq(|x| x.exp2(), self)
    }
    pub fn exp_m1(self) -> Sequence {
        apply1_seq(|x| x.exp_m1(), self)
    }
    pub fn floor(self) -> Sequence {
        apply1_seq(|x| x.floor(), self)
    }
    pub fn fract(self) -> Sequence {
        apply1_seq(|x| x.fract(), self)
    }
    pub fn hypot(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.hypot(y), self, other)
    }

    pub fn hypot_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.hypot(other), self)
    }

    pub fn is_finite(self) -> Sequence {
        apply1_seq(|x| fb(x.is_finite()), self)
    }
    pub fn is_infinite(self) -> Sequence {
        apply1_seq(|x| fb(x.is_infinite()), self)
    }
    pub fn is_nan(self) -> Sequence {
        apply1_seq(|x| fb(x.is_nan()), self)
    }
    pub fn is_normal(self) -> Sequence {
        apply1_seq(|x| fb(x.is_normal()), self)
    }
    pub fn is_sign_negative(self) -> Sequence {
        apply1_seq(|x| fb(x.is_sign_negative()), self)
    }
    pub fn is_sign_positive(self) -> Sequence {
        apply1_seq(|x| fb(x.is_sign_positive()), self)
    }
    pub fn is_subnormal(self) -> Sequence {
        apply1_seq(|x| fb(x.is_subnormal()), self)
    }

    pub fn ln(self) -> Sequence {
        apply1_seq(|x| x.ln(), self)
    }
    pub fn ln_1p(self) -> Sequence {
        apply1_seq(|x| x.ln_1p(), self)
    }
    pub fn log(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.log(y), self, other)
    }
    pub fn log_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.log(other), self)
    }
    pub fn log10(self) -> Sequence {
        apply1_seq(|x| x.log10(), self)
    }
    pub fn log2(self) -> Sequence {
        apply1_seq(|x| x.log2(), self)
    }
    pub fn max(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.max(y), self, other)
    }
    pub fn max_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.max(other), self)
    }
    pub fn maximum(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.maximum(y), self, other)
    }
    pub fn maximum_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.maximum(other), self)
    }
    pub fn midpoint(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.midpoint(y), self, other)
    }
    pub fn midpoint_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.midpoint(other), self)
    }
    pub fn min(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.min(y), self, other)
    }
    pub fn min_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.min(other), self)
    }

    pub fn minimum(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.minimum(y), self, other)
    }

    pub fn minimum_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.minimum(other), self)
    }
    pub fn next_down(self) -> Sequence {
        apply1_seq(|x| x.next_down(), self)
    }
    pub fn next_up(self) -> Sequence {
        apply1_seq(|x| x.next_up(), self)
    }
    pub fn powf(self, other: Self) -> Sequence {
        apply2_seq(|x, y| x.powf(y), self, other)
    }
    pub fn powf_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.powf(other), self)
    }
    pub fn recip(self) -> Sequence {
        apply1_seq(|x| x.recip(), self)
    }
    pub fn rem_euclid(self, other: Sequence) -> Sequence {
        apply2_seq(|x, y| x.rem_euclid(y), self, other)
    }
    pub fn rem_euclid_f(self, other: f64) -> Sequence {
        apply1_seq(move |x| x.rem_euclid(other), self)
    }
    pub fn round(self) -> Sequence {
        apply1_seq(|x| x.round(), self)
    }
    pub fn round_ties_even(self) -> Sequence {
        apply1_seq(|x| x.round_ties_even(), self)
    }
    pub fn signum(self) -> Sequence {
        apply1_seq(|x| x.signum(), self)
    }
    pub fn sin(self) -> Sequence {
        apply1_seq(|x| x.sin(), self)
    }
    pub fn sinh(self) -> Sequence {
        apply1_seq(|x| x.sinh(), self)
    }
    pub fn sqrt(self) -> Sequence {
        apply1_seq(|x| x.sqrt(), self)
    }
    pub fn tan(self) -> Sequence {
        apply1_seq(|x| x.tan(), self)
    }
    pub fn tanh(self) -> Sequence {
        apply1_seq(|x| x.tanh(), self)
    }
    pub fn to_degrees(self) -> Sequence {
        apply1_seq(|x| x.to_degrees(), self)
    }
    pub fn to_radians(self) -> Sequence {
        apply1_seq(|x| x.to_radians(), self)
    }
    pub fn geq(self, other: Sequence) -> Sequence {
        apply2_seq(move |x, y| if x >= y { 1.0 } else { 0.0 }, self, other)
    }
    pub fn geq_f(self, a: f64) -> Sequence {
        apply1_seq(move |x| if x >= a { 1.0 } else { 0.0 }, self)
    }
    pub fn leq(self, other: Sequence) -> Sequence {
        apply2_seq(move |x, y| if x <= y { 1.0 } else { 0.0 }, self, other)
    }
    pub fn leq_f(self, a: f64) -> Sequence {
        apply1_seq(move |x| if x <= a { 1.0 } else { 0.0 }, self)
    }
    pub fn mul(self, other: Sequence) -> Sequence {
        apply2_seq(move |x, y| x * y, self, other)
    }
    pub fn mul_f(self, a: f64) -> Sequence {
        apply1_seq(move |x| x * a, self)
    }

    pub fn add(self, other: Sequence) -> Sequence {
        apply2_seq(move |x, y| x + y, self, other)
    }
    pub fn add_f(self, a: f64) -> Sequence {
        apply1_seq(move |x| x + a, self)
    }
}

impl CustomType for Sequence {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("Sequence")
            .with_fn("seq", Sequence::new)
            .with_fn("diff", Sequence::diff)
            .with_fn("load", Sequence::load)
            .with_fn("distribution", Sequence::distribution)
            .with_fn("abs", Sequence::abs)
            .with_fn("acos", Sequence::acos)
            .with_fn("acosh", Sequence::acosh)
            .with_fn("asin", Sequence::asin)
            .with_fn("atan", Sequence::atan)
            .with_fn("atan2", Sequence::atan2)
            .with_fn("atan2", Sequence::atan2_f)
            .with_fn("cbrt", Sequence::cbrt)
            .with_fn("ceil", Sequence::ceil)
            .with_fn("clamp", Sequence::clamp)
            .with_fn("copysign", Sequence::copysign)
            .with_fn("copysign", Sequence::copysign_f)
            .with_fn("cos", Sequence::cos)
            .with_fn("cosh", Sequence::cosh)
            .with_fn("div_euclid", Sequence::div_euclid)
            .with_fn("div_euclid", Sequence::div_euclid_f)
            .with_fn("exp", Sequence::exp)
            .with_fn("exp2", Sequence::exp2)
            .with_fn("exp_m1", Sequence::exp_m1)
            .with_fn("floor", Sequence::floor)
            .with_fn("hypot", Sequence::hypot)
            .with_fn("hypot", Sequence::hypot_f)
            .with_fn("is_finite", Sequence::is_finite)
            .with_fn("is_infinite", Sequence::is_infinite)
            .with_fn("is_nan", Sequence::is_nan)
            .with_fn("is_normal", Sequence::is_normal)
            .with_fn("is_sign_negative", Sequence::is_sign_negative)
            .with_fn("is_sign_positive", Sequence::is_sign_positive)
            .with_fn("is_subnormal", Sequence::is_subnormal)
            .with_fn("ln", Sequence::ln)
            .with_fn("ln_1p", Sequence::ln_1p)
            .with_fn("log", Sequence::log)
            .with_fn("log", Sequence::log_f)
            .with_fn("log10", Sequence::log10)
            .with_fn("log2", Sequence::log2)
            .with_fn("max", Sequence::max)
            .with_fn("max", Sequence::max_f)
            .with_fn("maximum", Sequence::maximum)
            .with_fn("maximum", Sequence::maximum_f)
            .with_fn("midpoint", Sequence::midpoint)
            .with_fn("midpoint", Sequence::midpoint_f)
            .with_fn("min", Sequence::min)
            .with_fn("min", Sequence::min_f)
            .with_fn("minimum", Sequence::minimum)
            .with_fn("minimum", Sequence::minimum_f)
            .with_fn("next_down", Sequence::next_down)
            .with_fn("next_up", Sequence::next_up)
            .with_fn("powf", Sequence::powf)
            .with_fn("powf", Sequence::powf_f)
            .with_fn("recip", Sequence::recip)
            .with_fn("rem_euclid", Sequence::rem_euclid)
            .with_fn("rem_euclid", Sequence::rem_euclid_f)
            .with_fn("round", Sequence::round)
            .with_fn("round_ties_even", Sequence::round_ties_even)
            .with_fn("signum", Sequence::signum)
            .with_fn("sin", Sequence::sin)
            .with_fn("sinh", Sequence::sinh)
            .with_fn("sqrt", Sequence::sqrt)
            .with_fn("tan", Sequence::tan)
            .with_fn("tanh", Sequence::tanh)
            .with_fn("to_degrees", Sequence::to_degrees)
            .with_fn("to_radians", Sequence::to_radians)
            .with_fn("geq", Sequence::geq)
            .with_fn("geq", Sequence::geq_f)
            .with_fn("leq", Sequence::leq)
            .with_fn("leq", Sequence::leq_f)
            .with_fn("mul", Sequence::mul)
            .with_fn("mul", Sequence::mul_f);

        //     // Constructors
        //     .with_fn("constant", constant)
        //     .with_fn("uniform", uniform)
        //     .with_fn("uniform", uniform)
        //     .with_fn("uniform", uniform)
        //     .with_fn("normal", normal)
        //     .with_fn("bernoulli", bernoulli)
        //     .with_fn("poisson", poisson)
        //     .with_fn("exponential", exponential)
        //     // .with_fn("choice", choice)
        //     .with_fn("pareto", pareto)
        //     .with_fn("skew_normal", skew_normal)
        //     .with_fn("normal_range", normal_range)
        //     .with_fn("log_normal_range", log_normal_range)
        //     .with_fn("skewed_log_normal", skewed_log_normal)
        //     .with_fn(
        //         "log_normal_range_pct_deviation_at",
        //         log_normal_range_pct_deviation_at,
        //     );
        // // Operators
        // builder = build_bin_op(builder, "==", (|x, y| indicator(x == y)));
        // builder = build_bin_op(builder, "!=", (|x, y| indicator(x != y)));
        // builder = build_bin_op(builder, ">", (|x, y| indicator(x > y)));
        // builder = build_bin_op(builder, ">=", (|x, y| indicator(x >= y)));
        // builder = build_bin_op(builder, "<", (|x, y| indicator(x < y)));
        // builder = build_bin_op(builder, "<=", (|x, y| indicator(x <= y)));
        // builder = build_bin_op(
        //     builder,
        //     "&",
        //     (|x, y| indicator(deindicator(x) & deindicator(y))),
        // );
        // builder = build_bin_op(
        //     builder,
        //     "|",
        //     (|x, y| indicator(deindicator(x) | deindicator(y))),
        // );
        // builder = build_bin_op(builder, "^", (|x, y| x.powf(y)));
        // builder = build_bin_func(builder, "..", |x, y| uniform(x, y));
        // builder = build_bin_func(builder, "..=", |x, y| uniform(x, y));
        // builder = build_bin_op(builder, "+", (|x, y| x + y));
        // builder = build_bin_op(builder, "-", (|x, y| x - y));
        // builder = build_bin_op(builder, "*", (|x, y| x * y));
        // builder = build_bin_op(builder, "/", (|x, y| x / y));
        // builder = build_bin_op(builder, "%", (|x, y| x % y));
        // builder = build_bin_op(builder, "**", (|x, y| x.powf(y)));
        //
        // // Normal functions
        // builder
        //     .with_fn("!", a1(|x| 1.0 - x))
        //     .with_fn("abs", VariableExpr::abs)
        //     .with_fn("acos", VariableExpr::acos)
        //     .with_fn("acosh", VariableExpr::acosh)
        //     .with_fn("asin", VariableExpr::asin)
        //     .with_fn("atan", VariableExpr::atan)
        //     .with_fn("atan2", VariableExpr::atan2)
        //     .with_fn("atan2", VariableExpr::atan2_f)
        //     .with_fn("cbrt", VariableExpr::cbrt)
        //     .with_fn("ceil", VariableExpr::ceil)
        //     .with_fn("clamp", VariableExpr::clamp)
        //     .with_fn("copysign", VariableExpr::copysign)
        //     .with_fn("copysign", VariableExpr::copysign_f)
        //     .with_fn("cos", VariableExpr::cos)
        //     .with_fn("cosh", VariableExpr::cosh)
        //     .with_fn("div_euclid", VariableExpr::div_euclid)
        //     .with_fn("div_euclid", VariableExpr::div_euclid_f)
        //     .with_fn("exp", VariableExpr::exp)
        //     .with_fn("exp2", VariableExpr::exp2)
        //     .with_fn("exp_m1", VariableExpr::exp_m1)
        //     .with_fn("floor", VariableExpr::floor)
        //     .with_fn("hypot", VariableExpr::hypot)
        //     .with_fn("hypot", VariableExpr::hypot_f)
        //     .with_fn("is_finite", VariableExpr::is_finite)
        //     .with_fn("is_infinite", VariableExpr::is_infinite)
        //     .with_fn("is_nan", VariableExpr::is_nan)
        //     .with_fn("is_normal", VariableExpr::is_normal)
        //     .with_fn("is_sign_negative", VariableExpr::is_sign_negative)
        //     .with_fn("is_sign_positive", VariableExpr::is_sign_positive)
        //     .with_fn("is_subnormal", VariableExpr::is_subnormal)
        //     .with_fn("ln", VariableExpr::ln)
        //     .with_fn("ln_1p", VariableExpr::ln_1p)
        //     .with_fn("log", VariableExpr::log)
        //     .with_fn("log", VariableExpr::log_f)
        //     .with_fn("log10", VariableExpr::log10)
        //     .with_fn("log2", VariableExpr::log2)
        //     .with_fn("max", VariableExpr::max)
        //     .with_fn("max", VariableExpr::max_f)
        //     .with_fn("maximum", VariableExpr::maximum)
        //     .with_fn("maximum", VariableExpr::maximum_f)
        //     .with_fn("midpoint", VariableExpr::midpoint)
        //     .with_fn("midpoint", VariableExpr::midpoint_f)
        //     .with_fn("min", VariableExpr::min)
        //     .with_fn("min", VariableExpr::min_f)
        //     .with_fn("minimum", VariableExpr::minimum)
        //     .with_fn("minimum", VariableExpr::minimum_f)
        //     .with_fn("next_down", VariableExpr::next_down)
        //     .with_fn("next_up", VariableExpr::next_up)
        //     .with_fn("powf", VariableExpr::powf)
        //     .with_fn("powf", VariableExpr::powf_f)
        //     .with_fn("recip", VariableExpr::recip)
        //     .with_fn("rem_euclid", VariableExpr::rem_euclid)
        //     .with_fn("rem_euclid", VariableExpr::rem_euclid_f)
        //     .with_fn("round", VariableExpr::round)
        //     .with_fn("round_ties_even", VariableExpr::round_ties_even)
        //     .with_fn("signum", VariableExpr::signum)
        //     .with_fn("sin", VariableExpr::sin)
        //     .with_fn("sinh", VariableExpr::sinh)
        //     .with_fn("sqrt", VariableExpr::sqrt)
        //     .with_fn("tan", VariableExpr::tan)
        //     .with_fn("tanh", VariableExpr::tanh)
        //     .with_fn("to_degrees", VariableExpr::to_degrees)
        //     .with_fn("to_radians", VariableExpr::to_radians)
        //     .with_fn("geq", VariableExpr::geq)
        //     .with_fn("geq", VariableExpr::geq_f)
        //     .with_fn("leq", VariableExpr::leq)
        //     .with_fn("leq", VariableExpr::leq_f)
        //     .with_fn("mul", VariableExpr::mul)
        //     .with_fn("mul", VariableExpr::mul_f)
        //     // operators
        //     .with_fn(
        //         "lag",
        //         |base: &mut VariableExpr, lag_len: i64, init: VariableExpr| {
        //             lag(base.clone(), init, lag_len as usize)
        //         },
        //     )
        //     .with_fn("time", current_time);
    }
}
