pub fn fb(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

pub fn bent_line(slope1: f64, slope2: f64, sharpness: f64, x: f64) -> f64 {
    // w1 = (erf(sharpness * (x - kink)) + 1)/2
    // w2 =  1 - (erf(sharpness * (x - kink)) + 1)/2
    //    =  (1 - erf(sharpness * (x - kink)))/2

    let w1 = (erf(-sharpness * x) + 1.0) / 2.0;
    let w2 = 1.0 - w1;

    w1 * slope1 * x + w2 * slope2 * x
}

// y = a *w1(x) * x + b*w2(x)*x
//   = (a * w1 + b * w2) * x
// y/x = a * w1 + b (1 - w1)
//     = a*w1 + b - b*w1
//     = (a - b) * w1 +b
// r1 = (a - b) * w1 +b
//    = (a - b) * (erf(s*x1) + 1)/2 + b
// 2*r1 = (a - b) * (erf(s*x1) + 1) + 2 * b
// 2*r1 = (a - b) * erf(s*x1) + (a - b) + 2 * b

// 2*r1 = (a - b) * erf(s*x1) + a + b
// 2*r2 = (a - b) * erf(s*x2) + a + b
// assume s known
// 2*r1 = (1 + erf(s*x1)) * a + (1 - erf(s*x1)) * b
// 2*r2 = (1 + erf(s*x2)) * a + (1 - erf(s*x2)) * b
// =>
// w1 = c1 * a + c2 * b
// w2 = c3 * a + c4 * b
//
// c4 * w1 = c4 *  c1 * a + c4 * c2 * b
// c2 * w2 = c2 * c3 * a + c2 * c4 * b
// =>
// c4 * w1 - c2 * w2 = c4 *  c1 * a - c2 * c3 * a
// c4 * w1 - c2 * w2 = (c4 * c1 - c2 * c3) * a
// a = (c4 * w1 - c2 * w2) / (c4 * c1 - c2 * c3)
// AND
// b = (w2 - c3 * a) / c4

// assumes 0 -> 0
// x1, y1 < 0
// x2, y2 > 0
pub fn inverse_bent_line(
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    max_residual_contribution: f64,
) -> (f64, f64, f64) {
    // this criterion basically rules that the smoothing between the lines should be clamped at the control points
    // w1(x2) = (erf(-sharpness * x2) + 1.0)/2.0 < MRC;
    // w1(x2) = erf(-sharpness * x2) < 2.0 * MRC - 1.0;
    // -sharpness * x2 < erfinv(2.0 * MRC - 1.0);
    // sharpness < - erfinv(2.0 * MRC - 1.0) / x2;

    let min_sharpness_1 = -erf_inv(max_residual_contribution * 2.0 - 1.0) / x2;
    let min_sharpness_2 = -erf_inv(-(max_residual_contribution * 2.0 - 1.0)) / x1;

    let min_sharpness = min_sharpness_1.max(min_sharpness_2);

    let (a, b) = partial_inverse_bent_line(x1, x2, y1, y2, min_sharpness);

    (a, b, min_sharpness)
}

fn partial_inverse_bent_line(x1: f64, x2: f64, y1: f64, y2: f64, s: f64) -> (f64, f64) {
    let r1 = y1 / x1;
    let r2 = y2 / x2;

    let w1 = 2.0 * r1;
    let w2 = 2.0 * r2;
    let c1 = 1.0 + erf(-s * x1);
    let c2 = 1.0 - erf(-s * x1);
    let c3 = 1.0 + erf(-s * x2);
    let c4 = 1.0 - erf(-s * x2);

    let a = (c4 * w1 - c2 * w2) / (c4 * c1 - c2 * c3);
    let b = (w2 - c3 * a) / c4;

    (a, b)
}
