#![feature(float_minimum_maximum)]
#![feature(sort_floats)]

use {
    clap::ValueEnum,
    itertools::Itertools,
    parking_lot::RwLock,
    rhai::{Dynamic, Engine as RhaiEngine, LexError, Position},
    std::{
        collections::HashMap,
        iter,
        ops::{Range, RangeInclusive},
        sync::Arc,
    },
    textplots::{Chart, Plot, Shape, TickDisplay, TickDisplayBuilder},
};

mod distribution;
mod math;
mod variable;

pub use {distribution::*, math::*, variable::*};

#[derive(Debug, Clone, Hash, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab_case")]
pub enum DisplayMode {
    Text,
    Diagram,
}

impl TryFrom<&str> for DisplayMode {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "text" => Ok(DisplayMode::Text),
            "diagram" => Ok(DisplayMode::Diagram),
            _ => Err(format!("Invalid display mode: {}", s)),
        }
    }
}

impl Default for DisplayMode {
    fn default() -> Self {
        DisplayMode::Text
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Default)]
pub struct EngineOpts {
    pub display_mode: DisplayMode,
}

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
    opts: EngineOpts,
}

impl EngineCtx {
    pub fn new(opts: EngineOpts) -> Self {
        let mut rv = EngineCtx::default();
        rv.opts = opts;
        rv
    }
    pub fn variable_id_name(&self, id: u128) -> Option<String> {
        let ind = { self.variable_id_index.read().get(&id).cloned() };
        Some(self.variables.read().get(ind?)?.name.clone())
    }

    pub fn get_variable(&self, name: &str) -> Option<Variable> {
        let index = { self.variable_index.read().get(name)?.clone() };
        self.variables.read().get(index).cloned()
    }

    pub fn get_variable_value(&self, name: &str) -> Option<VariableExpr> {
        let index = { self.variable_index.read().get(name)?.clone() };
        Some(self.variables.read().get(index)?.posterior().clone())
    }
}

#[derive(Clone, Default)]
pub struct EngineRef {
    engine: Arc<EngineCtx>,
}

pub struct Engine {
    rhai_engine: RhaiEngine,
    ctx: EngineRef,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new(EngineOpts::default())
    }
}

impl Engine {
    pub fn new(engine_opts: EngineOpts) -> Engine {
        let mut engine = RhaiEngine::new();
        let ctx = EngineRef {
            engine: Arc::new(EngineCtx::new(engine_opts)),
        };

        engine
            .build_type::<VariableExpr>()
            .register_fn("report", Self::report_hook(ctx.clone()))
            .register_fn("simulate", Self::simulate_hook(ctx.clone()));

        Self::setup_prior_syntax(&mut engine, ctx.clone());
        Self::setup_p_syntax(&mut engine, ctx.clone());
        Self::setup_conditional_p_syntax(&mut engine, ctx.clone());
        Self::setup_expectation_syntax(&mut engine, ctx.clone());
        Self::setup_choice_syntax(&mut engine, ctx.clone());

        // Register the custom syntax: var x = ???

        Engine {
            rhai_engine: engine,
            ctx,
        }
    }

    fn dynamic_to_variable(value: Dynamic) -> Result<VariableExpr, String> {
        match value.type_name() {
            "i64" => Ok(constant(value.cast::<i64>() as f64)),
            "f64" => Ok(constant(value.cast::<f64>())),
            "decimal" => Ok(constant(value.cast::<f64>())),
            "core::ops::range::Range<i64>" => {
                let r: Range<i64> = value.cast();
                Ok(uniform(r.start as f64, r.end as f64))
            }
            "core::ops::range::Range<f64>" => {
                let r: Range<f64> = value.cast();
                Ok(uniform(r.start, r.end))
            }
            "core::ops::range::RangeInclusive<i64>" => {
                let r: RangeInclusive<i64> = value.cast();
                Ok(uniform(*r.start() as f64, *r.end() as f64))
            }
            "bool" => Ok(constant(indicator(value.cast::<bool>()))),
            "ruminant::variable::VariableExpr" => Ok(value.cast()),
            x => Err(format!("type {} cannot be used as a Random Variable", x)),
        }
    }

    fn dynamic_to_weight(value: Dynamic) -> Result<f64, String> {
        match value.type_name() {
            "i64" => Ok(value.cast::<i64>() as f64),
            "f64" => Ok(value.cast::<f64>()),
            "decimal" => Ok(value.cast::<f64>()),
            x => Err(format!("type {} cannot be used as a weight", x)),
        }
    }
    fn setup_prior_syntax(engine: &mut RhaiEngine, ctx: EngineRef) {
        engine
            .register_custom_syntax(
                ["prior", "$ident$", "=", "$expr$"],
                true,
                move |context, inputs| {
                    let var_name = inputs[0].get_string_value().unwrap().to_string();
                    let expr = &inputs[1];

                    // Evaluate the expression
                    let value = context.eval_expression_tree(expr)?;

                    let as_rv = Self::dynamic_to_variable(value)?;
                    let new_var = Self::make_variable(&ctx, var_name.clone(), as_rv);
                    // Push a new variable into the scope if it doesn't already exist.
                    // Otherwise just set its value.
                    if !context
                        .scope()
                        .is_constant(var_name.as_str())
                        .unwrap_or(false)
                    {
                        context.scope_mut().set_value(var_name.clone(), new_var);
                        Ok(Dynamic::UNIT)
                    } else {
                        Err(format!("variable {} is constant", var_name).into())
                    }
                },
            )
            .unwrap();
    }

    fn setup_p_syntax(engine: &mut RhaiEngine, ctx: EngineRef) {
        engine
            .register_custom_syntax(
                ["P", "[", "$ident$", "]", "=", "$expr$"],
                true,
                move |context, inputs| {
                    let var_name = inputs[0].get_string_value().unwrap().to_string();
                    let expr = &inputs[1];

                    // Evaluate the expression
                    let value = context.eval_expression_tree(expr)?;

                    let as_rv = Self::dynamic_to_variable(value)?;
                    // bernoulli-fy it
                    let as_rv = uniform(0.0, 1.0).leq(as_rv);

                    let new_var = Self::make_variable(&ctx, var_name.clone(), as_rv);
                    // Push a new variable into the scope if it doesn't already exist.
                    // Otherwise just set its value.
                    if !context
                        .scope()
                        .is_constant(var_name.as_str())
                        .unwrap_or(false)
                    {
                        context.scope_mut().set_value(var_name.clone(), new_var);
                        Ok(Dynamic::UNIT)
                    } else {
                        Err(format!("variable {} is constant", var_name).into())
                    }
                },
            )
            .unwrap();
    }

    fn make_variable(ctx: &EngineRef, name: String, value: VariableExpr) -> VariableExpr {
        let new_var = Variable::new(name.clone(), value, vec![]);
        let posterior = new_var.posterior();

        // store the rv
        let index = {
            let mut lock = ctx.engine.variables.write();
            lock.push(new_var);
            lock.len() - 1
        };

        // put a reference in the index
        {
            let mut lock = ctx.engine.variable_index.write();
            lock.insert(name.clone(), index);
        }

        {
            let mut lock = ctx.engine.variable_id_index.write();
            lock.insert(posterior.key, index);
        }

        posterior
    }

    fn setup_choice_syntax(engine: &mut RhaiEngine, ctx: EngineRef) {
        engine.register_custom_syntax_with_state_raw(
            // The leading symbol - which needs not be an identifier.
            "choice",
            // The custom parser implementation - always returns the next symbol expected
            // 'look_ahead' is the next symbol about to be read
            //
            // Return symbols starting with '$$' also terminate parsing but allows us
            // to determine which syntax variant was actually parsed so we can perform the
            // appropriate action.  This is a convenient short-cut to keeping the value
            // inside the state.
            //
            // The return type is 'Option<ImmutableString>' to allow common text strings
            // to be interned and shared easily, reducing allocations during parsing.
            |symbols, look_ahead, state| {
                if !symbols.is_empty() && symbols.last().unwrap() == "}" {
                    // terminal
                    return Ok(Some("$$choice".into()));
                }
                // choice ...
                if state.as_int().is_err() {
                    // state not yet init
                    if look_ahead.chars().next().unwrap() == '{' {
                        // no parent space
                        *state = 0.into();

                        return Ok(Some("{".into()));
                    } else {
                        // parent space
                        return Ok(Some("$ident$".into()));
                    }
                }

                let state_pos: i64 = state.as_int().unwrap();

                match (state_pos) % 4 {
                    0 => {
                        // increment state
                        *state = (state_pos + 1).into();

                        return Ok(Some("$ident$".into()));
                    }
                    1 => {
                        *state = (state_pos + 1).into();

                        return Ok(Some("=".into()));
                    }
                    2 => {
                        *state = (state_pos + 1).into();

                        return Ok(Some("$expr$".into()));
                    }
                    3 => {
                        *state = (state_pos + 1).into();

                        // either a comma or brace
                        match look_ahead.chars().next().unwrap() {
                            ',' => return Ok(Some(",".into())),
                            '}' => {
                                // terminal
                                return Ok(Some("}".into()));
                            }
                            x => {
                                return Err(LexError::ImproperSymbol(
                                    format!(
                                        "bad character {}, \
                            expected , or }}",
                                        x
                                    ),
                                    x.to_string(),
                                )
                                .into_err(Position::NONE));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            },
            // No variables declared/removed by this custom syntax
            true,
            // Implementation function
            move |context, inputs, _state| {
                let parent = if inputs[1].get_string_value().is_some() {
                    // if the first 2 are idents, then we have a parent cond
                    Some(
                        inputs[0]
                            .get_string_value()
                            .ok_or_else(|| {
                                "parent field in \
                    choice unspecified"
                                    .to_string()
                            })?
                            .to_string(),
                    )
                } else {
                    None
                };

                let mut scenarios = vec![];
                let mut read_ind = if parent.is_none() { 0 } else { 1 };
                // minus one because of the terminal char
                while read_ind < (inputs.len() - 1) {
                    let name = inputs[read_ind]
                        .get_string_value()
                        .ok_or_else(|| format!("bad identifier"))?
                        .to_string();
                    let expr = context.eval_expression_tree(&inputs[read_ind + 1])?;
                    let as_rv = Self::dynamic_to_variable(expr)?;

                    scenarios.push((name, as_rv));
                    read_ind += 2;
                }

                // done reading scenarios

                let active = if let Some(parent) = parent {
                    // there is a defined parent space
                    ctx.engine
                        .get_variable_value(parent.as_str())
                        .ok_or_else(|| {
                            format!(
                                "parent \
                   condition {} not defined",
                                parent
                            )
                        })?
                } else {
                    constant(1.0)
                };

                let partial_sums = scenarios.iter().fold(vec![], |mut acc, s| {
                    if acc.is_empty() {
                        acc.push(s.1.clone());
                    } else {
                        acc.push(acc.last().unwrap().clone().add(s.1.clone()));
                    }
                    acc
                });

                let likelihood_mass = partial_sums.last().unwrap().clone();

                let selector = uniform(0.0, 1.0).mul(likelihood_mass);

                for ((begin, end), scenario) in iter::once(constant(0.0))
                    .chain(partial_sums.into_iter())
                    .tuple_windows()
                    .zip(scenarios.into_iter())
                {
                    let scenario_var = selector
                        .clone()
                        .geq(begin)
                        .mul(selector.clone().leq(end))
                        .mul(active.clone());

                    let new_var = Self::make_variable(&ctx, scenario.0.clone(), scenario_var);

                    // put it into rhai context
                    context.scope_mut().set_value(scenario.0, new_var);
                }

                Ok(Dynamic::UNIT)
            },
        );
    }

    fn setup_conditional_p_syntax(engine: &mut RhaiEngine, ctx: EngineRef) {
        engine
            .register_custom_syntax(
                [
                    "weight", "$expr$", "P", "[", "$ident$", "|", "$expr$", "]", "=", "$expr$",
                ],
                false,
                move |context, inputs| {
                    let weight = &inputs[0];
                    let var_name = inputs[1].get_string_value().unwrap().to_string();
                    let condition = &inputs[2];
                    let expr = &inputs[3];

                    let weight_value = context.eval_expression_tree(weight)?;
                    let weight_value = Self::dynamic_to_weight(weight_value)?;

                    // Evaluate the condition
                    let cond_value = context.eval_expression_tree(condition)?;
                    let cond_rv = Self::dynamic_to_variable(cond_value)?;
                    // bernoulli-fy it
                    let cond_rv = uniform(0.0, 1.0).leq(cond_rv);

                    // Evaluate the expression
                    let value = context.eval_expression_tree(expr)?;
                    let as_rv = Self::dynamic_to_variable(value)?;
                    // bernoulli-fy it
                    let as_rv = uniform(0.0, 1.0).leq(as_rv);

                    // this actually does nothing in rhai-world
                    let index = ctx
                        .engine
                        .variable_index
                        .read()
                        .get(var_name.as_str())
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "conditionally supplied for variable {} without a prior. make \
                        sure to specify a prior before giving conditionals",
                                var_name
                            )
                        })?;

                    // put a reference in the index
                    {
                        let _lock = ctx
                            .engine
                            .variables
                            .write()
                            .get_mut(index)
                            .unwrap()
                            .add_conditional(Conditional {
                                weight: Some(weight_value),
                                condition: cond_rv,
                                value: as_rv,
                            });
                    }

                    Ok(Dynamic::UNIT)
                },
            )
            .unwrap();
    }

    fn setup_expectation_syntax(engine: &mut RhaiEngine, ctx: EngineRef) {
        engine
            .register_custom_syntax(
                [
                    "E", "[", "$ident$", "|", "$expr$", "]", "[", "$expr$", "]", "=", "$expr$",
                ],
                false,
                move |context, inputs| {
                    let weight = &inputs[2];
                    let var_name = inputs[0].get_string_value().unwrap().to_string();
                    let condition = &inputs[1];
                    let expr = &inputs[3];

                    let weight_value = context.eval_expression_tree(weight)?;
                    let weight_value = Self::dynamic_to_weight(weight_value)?;

                    // Evaluate the condition
                    let cond_value = context.eval_expression_tree(condition)?;
                    let cond_rv = Self::dynamic_to_variable(cond_value)?;
                    // bernoulli-fy it
                    let cond_rv = uniform(0.0, 1.0).leq(cond_rv);

                    // Evaluate the expression
                    let value = context.eval_expression_tree(expr)?;
                    let as_rv = Self::dynamic_to_variable(value)?;

                    // this actually does nothing in rhai-world
                    let index = ctx
                        .engine
                        .variable_index
                        .read()
                        .get(var_name.as_str())
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "conditional expectation supplied for variable {} without a prior. \
                                make sure to specify a prior before giving conditionals",
                                var_name
                            )
                        })?;

                    // put a reference in the index
                    {
                        let _lock = ctx
                            .engine
                            .variables
                            .write()
                            .get_mut(index)
                            .unwrap()
                            .add_conditional(Conditional {
                                weight: Some(weight_value),
                                condition: cond_rv,
                                value: as_rv,
                            });
                    }

                    Ok(Dynamic::UNIT)
                },
            )
            .unwrap();
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
                timestep: 0,
                values: values.clone(),
            };

            match ctx.engine.opts.display_mode {
                DisplayMode::Text => {
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
                DisplayMode::Diagram => {
                    // We have to bucketify the value if we want to display it as a diagram
                    // first, we figure out what is a reasonable bucket size based on the range of the data
                    // then, make that many buckets and fill them with the values
                    // defaulting to 15 buckets for now
                    //
                    let two_std_left = report.quantile(0.05);
                    let two_std_right = report.quantile(0.95);
                    let bucket_size = (two_std_right - two_std_left) / 15.0;

                    let mut buckets = vec![0; 15];

                    for value in &values {
                        if value <= &two_std_left || value >= &two_std_right {
                            continue;
                        }

                        let bucket_index = ((value - two_std_left) / bucket_size) as usize;
                        buckets[bucket_index] += 1;
                    }

                    let points = buckets
                        .into_iter()
                        .enumerate()
                        .map(|(i, count)| {
                            let x = (two_std_left + i as f64 * bucket_size) as f32;
                            let y = count as f32;
                            (x, y)
                        })
                        .collect::<Vec<_>>();

                    println!("Variable: {}", name);
                    Chart::new(180, 60, two_std_left as f32, two_std_right as f32)
                        .lineplot(&Shape::Bars(&points))
                        .display();
                }
            }
        }
    }

    pub fn simulate_hook(ctx: EngineRef) -> impl Fn(VariableExpr, i64) {
        move |var, n_steps| {
            let ctx = ctx.clone();
            let id = var.key;

            let name = ctx
                .engine
                .variable_id_name(id)
                .unwrap_or_else(|| format!("{}", id));

            let mut reports = vec![];

            for timestep in 0..n_steps {
                let values = var
                    .eval_n(&ctx, timestep as usize, 10000)
                    .map_err(|e| format!("error sampling {}:{}", name, e))
                    .unwrap();

                let report = Report {
                    name: name.clone(),
                    timestep: timestep as usize,
                    values: values.clone(),
                };

                reports.push(report);
            }

            match ctx.engine.opts.display_mode {
                DisplayMode::Text => {
                    for report in reports {
                        println!(
                            "{{\"variable\": \"{}\", \"time\": {}, \"mean\": {}, \"std\":{}, \
                            \"q10\":{},\"q25\":{},\
                        \"q50\":{},\"q75\":{},\"q90\":{}}}",
                            name,
                            report.timestep,
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
                DisplayMode::Diagram => {
                    let q10: Vec<f64> = reports.iter().map(|r| r.quantile(0.1)).collect();
                    let q25: Vec<f64> = reports.iter().map(|r| r.quantile(0.25)).collect();
                    let q50: Vec<f64> = reports.iter().map(|r| r.quantile(0.5)).collect();
                    let q75: Vec<f64> = reports.iter().map(|r| r.quantile(0.75)).collect();
                    let q90: Vec<f64> = reports.iter().map(|r| r.quantile(0.9)).collect();

                    println!("Variable: {}", name);
                    Chart::new(180, 60, 0.0, reports.len() as f32)
                        // .x_tick_display(TickDisplay::Sparse)
                        .y_tick_display(TickDisplay::Sparse)
                        .lineplot(&Shape::Continuous(Box::new(|x| {
                            q10[x.floor() as usize] as f32
                        })))
                        .lineplot(&Shape::Continuous(Box::new(|x| {
                            q25[x.floor() as usize] as f32
                        })))
                        .lineplot(&Shape::Continuous(Box::new(|x| {
                            q50[x.floor() as usize] as f32
                        })))
                        .lineplot(&Shape::Continuous(Box::new(|x| {
                            q75[x.floor() as usize] as f32
                        })))
                        .lineplot(&Shape::Continuous(Box::new(|x| {
                            q90[x.floor() as usize] as f32
                        })))
                        .display();
                }
            }
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
            timestep: 0,
            values,
        })
    }
}

pub struct Report {
    pub name: String,
    pub timestep: usize,
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
        let mut engine = Engine::default();
        let source = "prior u1 = uniform(0.0, 1.0);
        prior u2 = 2.0;
        prior u3 = true;
        prior u4 = 2;
        prior u5 = 1..7;
        prior u6 = u1 + u2;

        report(u6);";

        engine.run(source).unwrap();

        assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }

    #[test]
    fn p_syntax() {
        let mut engine = Engine::default();
        let source = "\
        prior u1 = uniform(0.0, 1.0);

        P[some_event] = 0.5;
        P[some_other_event] = u1;

        report(some_event);
        report(some_other_event);
        ";

        engine.run(source).unwrap();

        // assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }

    #[test]
    fn choice_syntax() {
        let mut engine = Engine::default();
        let source = include_str!("../examples/choice_syntax.rm");

        engine.run(source).unwrap();

        // assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }
    #[test]
    fn conditional_p_syntax() {
        let mut engine = Engine::default();
        let source = include_str!("../examples/conditional_probability.rm");

        engine.run(source).unwrap();

        // assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }

    #[test]
    fn conditional_expectation_syntax() {
        let mut engine = Engine::default();
        let source = include_str!("../examples/conditional_expectation.rm");

        engine.run(source).unwrap();

        // assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }

    #[test]
    fn timeseries() {
        let mut engine = Engine::default();
        let source = include_str!("../examples/time_series.rm");

        engine.run(source).unwrap();

        // assert!((engine.report("u6").unwrap().mean() - 2.5).abs() < 0.1);
    }
}
