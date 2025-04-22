#[derive(Debug, Clone)]
pub struct Centroid {
    pub mean: f64,
    pub weight: f64,
    pub sort_key1: isize,
    pub sort_key2: isize,
}

impl Centroid {
    pub fn to_string(&self) -> String {
        format!(
            "{{\"mean\": \"{mean}\",\"weight\": \"{weight}\"}}",
            mean = self.mean,
            weight = self.weight
        )
    }

    pub fn add(&mut self, r: &Centroid) -> String {
        if r.weight < 0.0 {
            return "centroid weight cannot be less than zero".to_string();
        }
        if self.weight != 0.0 {
            self.weight += r.weight;
            self.mean += r.weight * (r.mean - self.mean) / self.weight;
        } else {
            self.weight = r.weight;
            self.mean = r.mean;
        }
        "".to_string()
    }
}
