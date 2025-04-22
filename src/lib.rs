#![feature(float_minimum_maximum)]

use parking_lot::RwLock;
use rhai::{CustomType, Dynamic, Engine as RhaiEngine, EvalAltResult, TypeBuilder};
use std::collections::HashMap;
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
    particles: RwLock<HashMap<ParticleKey, Particle>>,
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
            .register_fn("report", Self::report(ctx.clone()));

        // Register the custom syntax: var x = ???
        // engine.register_custom_syntax(
        //     ["var", "$ident$", "=", "$expr$"],
        //     true,
        //     |context, inputs| {
        //         let var_name = inputs[0].get_string_value().unwrap().to_string();
        //         let expr = &inputs[1];
        //
        //         // Evaluate the expression
        //         let value = context.eval_expression_tree(expr)?;
        //
        //         // Push a new variable into the scope if it doesn't already exist.
        //         // Otherwise just set its value.
        //         if !context.scope().is_constant(var_name).unwrap_or(false) {
        //             context.scope_mut().set_value(var_name.to_string(), value);
        //             Ok(Dynamic::UNIT)
        //         } else {
        //             Err(format!("variable {} is constant", var_name).into())
        //         }
        //     },
        // )?;

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

    pub fn report(ctx: EngineRef) -> impl Fn(VariableExpr) {
        move |var| {
            let ctx = ctx.clone();
            let id = var.key;
            let samples: Vec<f64> = (0..10000)
                .map(move |n| var.eval(&ctx, n, 0).unwrap())
                .collect();
            let mean = samples.iter().fold(0.0, |acc, v| acc + v) / samples.len() as f64;
            println!("Variable {{{}}} mean {{{}}}", id, mean);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raincoat() {
        let mut engine = Engine::new();
        let source = include_str!("examples/raincoat.rm");

        engine.run(source).unwrap();
    }
}
