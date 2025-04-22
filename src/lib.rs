#![feature(float_minimum_maximum)]
#![feature(sort_floats)]

use parking_lot::RwLock;
use rhai::{CustomType, Dynamic, Engine as RhaiEngine, EvalAltResult, TypeBuilder};
use std::collections::HashMap;
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;
use uuid::uuid;

mod distribution;
mod math;
mod variable;

pub use distribution::*;
pub use math::*;
pub use variable::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ParticleKey {
    pub generation: usize,
    pub timestep: usize,
    pub variable: u128,
}

pub struct Particle {
    pub value: f64,
}

pub struct Scenario {
    pub indicator: usize,
}

#[derive(Default)]
pub struct EngineCtx {
    variables: RwLock<Vec<Variable>>,
    variable_index: RwLock<HashMap<String, usize>>,
    variable_id_index: RwLock<HashMap<u128, usize>>,
    particles: RwLock<HashMap<ParticleKey, Particle>>,
}

impl EngineCtx {
    pub fn variable_id_name(&self, id: u128) -> Option<String> {
        let ind = { self.variable_id_index.read().get(&id).cloned() };
        Some(self.variables.read().get(ind?)?.name.clone())
    }

    pub fn get_variable(&self, name: &str) -> Option<Variable> {
        let index = { self.variable_index.read().get(name)?.clone() };
        self.variables.read().get(index).cloned()
    }
}

#[derive(Clone)]
pub struct EngineRef {
    engine: Arc<EngineCtx>,
}

pub struct Engine {
    rhai_engine: RhaiEngine,
    ctx: EngineRef,
}

impl Engine {
    pub fn new() -> Engine {
        let mut engine = RhaiEngine::new();
        let ctx = EngineRef {
            engine: Arc::new(EngineCtx::default()),
        };

        engine
            .build_type::<VariableExpr>()
            .register_fn("report", Self::report_hook(ctx.clone()));

        let ctx2 = ctx.clone();
        // Register the custom syntax: var x = ???
        engine
            .register_custom_syntax(
                ["var", "$ident$", "=", "$expr$"],
                true,
                move |context, inputs| {
                    let var_name = inputs[0].get_string_value().unwrap().to_string();
                    let expr = &inputs[1];

                    // Evaluate the expression
                    let value = context.eval_expression_tree(expr)?;

                    let as_rv: VariableExpr = match value.type_name() {
                        "i64" => constant(value.cast::<i64>() as f64),
                        "f64" => constant(value.cast::<f64>()),
                        "decimal" => constant(value.cast::<f64>()),
                        "core::ops::range::Range<i64>" => {
                            let r: Range<i64> = value.cast();
                            uniform(r.start as f64, r.end as f64)
                        }
                        "core::ops::range::Range<f64>" => {
                            let r: Range<f64> = value.cast();
                            uniform(r.start, r.end)
                        }
                        "core::ops::range::RangeInclusive<i64>" => {
                            let r: RangeInclusive<i64> = value.cast();
                            uniform(*r.start() as f64, *r.end() as f64)
                        }
                        "bool" => constant(indicator(value.cast::<bool>())),
                        "ruminant::variable::VariableExpr" => value.cast(),
                        x => panic!("type {} cannot be used as a Random Variable", x),
                    };

                    let key = as_rv.key;

                    // Push a new variable into the scope if it doesn't already exist.
                    // Otherwise just set its value.
                    if !context
                        .scope()
                        .is_constant(var_name.as_str())
                        .unwrap_or(false)
                    {
                        context
                            .scope_mut()
                            .set_value(var_name.clone(), as_rv.clone());

                        // store the rv
                        let index = {
                            let mut lock = ctx2.engine.variables.write();
                            lock.push(Variable::new(var_name.clone(), as_rv));
                            lock.len() - 1
                        };

                        // put a reference in the index
                        {
                            let mut lock = ctx2.engine.variable_index.write();
                            lock.insert(var_name.clone(), index);
                        }

                        {
                            let mut lock = ctx2.engine.variable_id_index.write();
                            lock.insert(key, index);
                        }

                        Ok(Dynamic::UNIT)
                    } else {
                        Err(format!("variable {} is constant", var_name).into())
                    }
                },
            )
            .unwrap();

        Engine {
            rhai_engine: engine,
            ctx,
        }
    }

    pub fn run(&mut self, script: &str) -> Result<(), String> {
        Ok(self
            .rhai_engine
            .run(script)
            .map_err(|e| format!("{:?}", e))?)
    }

    pub fn report_hook(ctx: EngineRef) -> impl Fn(VariableExpr) {
        move |var| {
            let ctx = ctx.clone();
            let id = var.key;

            let name = ctx
                .engine
                .variable_id_name(id)
                .unwrap_or_else(|| format!("{}", id));

            let values = var
                .eval_n(&ctx, 0, 10000)
                .map_err(|e| format!("error sampling {}:{}", name, e))
                .unwrap();

            let report = Report {
                name: name.clone(),
                values,
            };

            println!(
                "{{\"variable\": \"{}\", \"mean\": {}, \"std\":{}, \"q10\":{},\"q25\":{},\
            \"q50\":{},\"q75\":{},\"q90\":{}}}",
                name,
                report.mean(),
                report.std(),
                report.quantile(0.1),
                report.quantile(0.25),
                report.quantile(0.5),
                report.quantile(0.75),
                report.quantile(0.9)
            );
        }
    }

    pub fn report(&self, var: &str) -> Option<Report> {
        let values = self
            .ctx
            .engine
            .get_variable(var)?
            .eval_n(&self.ctx, 0, 10000)
            .ok()?;
        Some(Report {
            name: var.to_string(),
            values,
        })
    }
}

pub struct Report {
    pub name: String,
    pub values: Vec<f64>,
}

impl Report {
    pub fn mean(&self) -> f64 {
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    pub fn std(&self) -> f64 {
        let mean = self.mean();
        (self
            .values
            .iter()
            .fold(0.0, |acc, v| acc + (v - mean).powf(2.0))
            / self.values.len() as f64)
            .sqrt()
    }
    pub fn quantile(&self, q: f64) -> f64 {
        let mut v2 = self.values.clone();
        v2.sort_floats();
        v2[(q * v2.len() as f64).floor() as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn var_syntax() {
        let mut engine = Engine::new();
        let source = "\
        var u1 = uniform(0.0, 1.0);
        var u2 = 2.0;
        var u3 = true;
        var u4 = 2;
        var u5 = 1..7;
        var u6 = u1 + u2;

        report(u6);
        ";

        engine.run(source).unwrap();

        assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }

    #[test]
    fn raincoat() {
        let mut engine = Engine::new();
        let source = include_str!("examples/raincoat.rm");

        engine.run(source).unwrap();
    }
}
