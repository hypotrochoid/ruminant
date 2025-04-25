use {
    crate::{VariableExpr, tdigest::Tdigest},
    itertools::Itertools,
    rand::Rng,
    rhai::{Array, CustomType, Dynamic, TypeBuilder},
};

#[derive(Clone, Default, Debug)]
pub struct EmpiricalDist {
    dist: Tdigest,
}

impl EmpiricalDist {
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

        let mut dist = Tdigest::default();
        for val in v? {
            dist.add(val, 1.0);
        }
        Ok(EmpiricalDist { dist })
    }

    pub fn insert(&mut self, value: f64) {
        self.dist.add(value, 1.0);
    }

    pub fn distribution(mut self) -> VariableExpr {
        // ref inside closure can't be mutatble
        self.dist.process();
        VariableExpr::new(move |engine, generation, timestep| {
            let sel: f64 = rand::rng().random();
            Ok(self.dist.quantile_processed(sel))
        })
    }
}

impl CustomType for EmpiricalDist {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("EmpiricalDist")
            .with_fn("insert", EmpiricalDist::insert)
            .with_fn("distribution", EmpiricalDist::distribution);
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
