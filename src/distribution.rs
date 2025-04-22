use super::math::*;
use super::{VariableExpr, a1, a2, apply1};
use rand::Rng;
use rand_distr::Distribution;
use statrs::function::erf::erf_inv;

#[derive(Clone, Debug)]
pub enum DistributionVariable {
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std: f64 },
    Bernoulli { p: f64 },
    Poisson { arrival_rate: f64 },
    Exponential { scale: f64 },
    Constant(f64),
    Choice(Vec<f64>),
    WeightedChoice(Vec<(f64, f64)>),
    Pareto { scale: f64, shape: f64 },
}

impl DistributionVariable {
    pub fn inf(&self) -> f64 {
        match self {
            DistributionVariable::Uniform { min, .. } => *min,
            DistributionVariable::Bernoulli { .. } => 0.0,
            DistributionVariable::Poisson { .. } => 0.0,
            DistributionVariable::Exponential { .. } => 0.0,
            DistributionVariable::Constant(x) => *x,
            DistributionVariable::Choice(x) => {
                *x.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            }
            DistributionVariable::WeightedChoice(x) => *x
                .iter()
                .map(|(x, y)| x)
                .min_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap(),
            DistributionVariable::Pareto { scale, .. } => *scale,
            _ => f64::NEG_INFINITY,
        }
    }

    pub fn sup(&self) -> f64 {
        match self {
            DistributionVariable::Uniform { max, .. } => *max,
            DistributionVariable::Bernoulli { .. } => 1.0,
            DistributionVariable::Poisson { .. } => f64::INFINITY,
            DistributionVariable::Exponential { .. } => f64::INFINITY,
            DistributionVariable::Constant(x) => *x,
            DistributionVariable::Choice(x) => {
                *x.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            }
            DistributionVariable::WeightedChoice(x) => *x
                .iter()
                .map(|(x, y)| x)
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap(),
            DistributionVariable::Pareto { .. } => f64::INFINITY,
            _ => f64::INFINITY,
        }
    }

    fn sample_n(&self, mut rng: &mut impl Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(&mut rng)).collect()
    }
    fn sample(&self, mut rng: &mut impl Rng) -> f64 {
        match self {
            DistributionVariable::Uniform { min, max } => {
                rand_distr::Uniform::new(min, max).unwrap().sample(rng)
            }
            DistributionVariable::Normal { mean, std } => {
                rand_distr::Normal::new(*mean, *std).unwrap().sample(rng)
            }
            DistributionVariable::Bernoulli { p } => {
                if rand_distr::Bernoulli::new(*p).unwrap().sample(rng) {
                    1.0
                } else {
                    0.0
                }
            }
            DistributionVariable::Poisson { arrival_rate } => {
                rand_distr::Poisson::new(*arrival_rate).unwrap().sample(rng)
            }
            DistributionVariable::Exponential { scale } => {
                rand_distr::Exp::new(*scale).unwrap().sample(rng)
            }
            DistributionVariable::Constant(x) => *x,
            DistributionVariable::Pareto { scale, shape } => {
                rand_distr::Pareto::new(*scale, *shape).unwrap().sample(rng)
            }
            DistributionVariable::Choice(choices) => {
                let ind = rng.random_range(0..choices.len());
                choices[ind].clone()
            }
            DistributionVariable::WeightedChoice(choices) => {
                let sel = rand_distr::Uniform::new(0.0, 1.0).unwrap().sample(rng);
                let mut csum = 0.0;
                for (weight, val) in choices.iter() {
                    csum += *weight;
                    if csum >= sel {
                        return *val;
                    }
                }
                choices.last().unwrap().1.clone()
            }
        }
    }

    fn mean(&self) -> f64 {
        match self {
            DistributionVariable::Uniform { min, max } => min.midpoint(*max),
            DistributionVariable::Normal { mean, std } => *mean,
            DistributionVariable::Bernoulli { p } => *p,
            DistributionVariable::Poisson { arrival_rate } => *arrival_rate,
            DistributionVariable::Exponential { scale } => scale.powf(-1.0),
            DistributionVariable::Constant(x) => *x,
            DistributionVariable::Pareto { scale, shape } => {
                if *shape <= 1.0 {
                    f64::INFINITY
                } else {
                    shape * scale / (shape - 1.0)
                }
            }
            _ => {
                self.sample_n(&mut rand::rng(), 10000)
                    .iter()
                    .fold(0_f64, |acc, x| acc + *x)
                    / 10000.0
            }
        }
    }

    fn std(&self) -> f64 {
        match self {
            DistributionVariable::Uniform { min, max } => (max - min).abs() / 12.0_f64.sqrt(),
            DistributionVariable::Normal { mean, std } => *std,
            DistributionVariable::Bernoulli { p } => (p * (1.0 - p)).sqrt(),
            DistributionVariable::Poisson { arrival_rate } => *arrival_rate,
            DistributionVariable::Exponential { scale } => scale.powf(-1.0),
            DistributionVariable::Constant(x) => 0.0,
            _ => {
                let mean = self.mean();
                (self
                    .sample_n(&mut rand::rng(), 10000)
                    .iter()
                    .fold(0_f64, |acc, x| acc + (*x - mean).powf(2.0))
                    / 10000_f64)
                    .sqrt()
            }
        }
    }

    pub fn sampler(self) -> VariableExpr {
        VariableExpr::new(move |engine, generation, t| self.sample(&mut rand::rng()))
    }
}

// ergonomic functions
pub fn constant(value: f64) -> VariableExpr {
    DistributionVariable::Constant(value).sampler()
}
pub fn uniform(min: f64, max: f64) -> VariableExpr {
    DistributionVariable::Uniform { min, max }.sampler()
}

pub fn normal(mean: f64, std: f64) -> VariableExpr {
    DistributionVariable::Normal { mean, std }.sampler()
}

pub fn bernoulli(p: f64) -> VariableExpr {
    DistributionVariable::Bernoulli { p }.sampler()
}

pub fn poisson(arrival_rate: f64) -> VariableExpr {
    DistributionVariable::Poisson { arrival_rate }.sampler()
}

pub fn exponential(scale: f64) -> VariableExpr {
    DistributionVariable::Exponential { scale }.sampler()
}

pub fn choice(choices: Vec<f64>) -> VariableExpr {
    DistributionVariable::Choice(choices).sampler()
}

pub fn weighted_choice(choices: Vec<(f64, f64)>) -> VariableExpr {
    DistributionVariable::WeightedChoice(choices).sampler()
}

pub fn pareto(median: f64, mode: f64) -> VariableExpr {
    let scale = mode;
    let shape = ((median / scale).ln() / 2_f64.ln()).powf(-1.0);

    DistributionVariable::Pareto { scale, shape }.sampler()
}

pub fn skew_normal(q_10: f64, q_50: f64, q_90: f64) -> VariableExpr {
    let upper_dist = q_90 - q_50;
    let lower_dist = q_10 - q_50;

    let max_residual_contribution = 1e-4;
    let min_sharpness_1 = -erf_inv(max_residual_contribution * 2.0 - 1.0) / upper_dist;
    let min_sharpness_2 = -erf_inv(-(max_residual_contribution * 2.0 - 1.0)) / lower_dist;

    let s = min_sharpness_1.max(min_sharpness_2);

    let a = -0.8 * lower_dist;
    let b = 0.8 * upper_dist;

    apply1(
        move |x| bent_line(a, b, s, x) + q_50,
        DistributionVariable::Normal {
            mean: 0.0,
            std: 1.0,
        }
        .sampler(),
    )
}

// 4 sigma range normal clamped at the limits
pub fn normal_range(min: f64, max: f64) -> VariableExpr {
    let mean = min.midpoint(max);
    let std = (mean - min) / 2.0;
    normal(mean, std).clamp(min, max)
}

// 4 sigma log range normal clamped at the limits
pub fn log_normal_range(min: f64, max: f64) -> VariableExpr {
    let log_min = min.ln();
    let log_max = max.ln();

    let underlying_mean = log_min.midpoint(log_max);
    let underlying_std = (underlying_mean - log_min) / 2.0;

    let ex = normal(underlying_mean, underlying_std).exp();
    ex.clamp(min, max)
}

pub fn skewed_log_normal(q_10: f64, q_50: f64, q_90: f64) -> VariableExpr {
    let log_q10 = q_10.ln();
    let log_q50 = q_50.ln();
    let log_q90 = q_90.ln();

    skew_normal(log_q10, log_q50, log_q90).exp()
}

// pub fn mixture_model(mut models: Vec<(VariableExpr, f64)>) -> VariableExpr {
//     let init = models.pop().unwrap();
//     let mut total_weight = init.1;
//     let mut retv = init.0.mul(init.1);
//     for nrv in models.into_iter() {
//         total_weight += nrv.1;
//         retv = (retv + nrv.0.mul(nrv.1));
//     }
//
//     retv.mul(1.0 / total_weight)
// }

// 4 sigma log range normal clamped at the limits
pub fn log_normal_range_pct_deviation_at(center: f64, pct_dev: f64) -> VariableExpr {
    let min = center.clone() * (1.0 - pct_dev.clone());
    let max = center.clone() * (1.0 + pct_dev.clone());

    let log_min = min.ln();
    let log_max = max.ln();

    let underlying_mean = log_min.midpoint(log_max);
    let underlying_std = (underlying_mean - log_min) / 2.0;

    normal(underlying_mean, underlying_std)
        .exp()
        .clamp(min, max)
}

// a = (y2 - y1)/ (x2 - x1)
// b = y1 - x1 * a
// linear interpolation of RVs
// pub fn linterp(x: VariableExpr, control_points: Vec<(f64, VariableExpr)>) -> VariableExpr {
//     let mut breaks = vec![];
//     for pair in 0..(control_points.len() - 1) {
//         let x1 = rv(control_points[pair].0.clone());
//         let x1_2 = rv(control_points[pair].0.clone());
//         let x2 = rv(control_points[pair + 1].0.clone());
//         let y1 = (control_points[pair + 1].1).clone();
//         let y1_2 = (control_points[pair + 1].1).clone();
//         let y2 = (control_points[pair + 1].1).clone();
//
//         let slope = ((y2 - y1) / (x2 - x1)).sampler();
//         let intersect = y1_2 - (x1_2 * slope.clone());
//
//         let resp = (slope * x.clone()) + intersect;
//         breaks.push((control_points[pair].0.clone(), resp))
//     }
//
//     piecewise_variable(x, breaks)
// }

// a / x^b = y
// ln(a) - b ln(x) = ln(y)
//
// ln(a) - b ln(x1) = ln(y1)
// ln(a) - b ln(x2) = ln(y2)
//
// - b ln(x1) + b ln(x2)= ln(y1) - ln(y2)
// b = (ln(y1) - ln(y2)) / (ln(x2) - ln(x1) )
//   = (ln((y1 / y2).powf(-ln(x2/x1)))
//   =
// ln(a) = ln(y1) + b ln(x1)
//       = ln (y1 * x1^b)
// a = y1*x1^b

// pub fn power_law(x: VariableExpr, control_points: Vec<(f64, VariableExpr)>) -> VariableExpr {
//     let mut breaks = vec![];
//     for pair in 0..(control_points.len() - 1) {
//         let x1 = control_points[pair].0.clone();
//         let x2 = control_points[pair].0.clone();
//         let y1 = (control_points[pair + 1].1).clone();
//         let y1_2 = (control_points[pair + 1].1).clone();
//         let y2 = (control_points[pair + 1].1).clone();
//
//         let b = (y1 / y2)
//             .sampler()
//             .powf((-rv(x2 / x1).ln()).sampler())
//             .ln()
//             .sampler();
//         let a = (y1_2 * (rv(x1).powf(b.clone()))).sampler();
//
//         let resp = a / (x.clone().powf(b));
//         let mp = control_points[pair + 1].0.clone();
//         breaks.push((mp, resp.sampler()))
//     }
//
//     piecewise_variable(x, breaks)
// }
