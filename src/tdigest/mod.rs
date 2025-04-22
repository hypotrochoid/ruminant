mod centroid;

use {
    centroid::Centroid,
    serde_json::{json, Value},
    std::{
        fs::File,
        io::{BufRead, BufReader},
    },
};

#[derive(Debug, Clone, Default)]
pub struct Tdigest {
    compression: f64,
    max_processed: usize,
    max_unprocessed: usize,
    processed: Vec<Centroid>,
    unprocessed: Vec<Centroid>,
    cumulative: Vec<f64>,
    processed_weight: f64,
    unprocessed_weight: f64,
    min: f64,
    max: f64,
    variance: f64,
    count: f64,
    sum: f64,
}

impl Tdigest {
    // For the compression parameter.  100 is a common value for normal uses.  1000 is extremely large.
    pub fn new(set_compression: f64) -> Tdigest {
        Tdigest {
            compression: set_compression,
            max_processed: processed_size(0, set_compression),
            max_unprocessed: unprocessed_size(0, set_compression),
            processed: Vec::with_capacity(2 * set_compression as usize),
            unprocessed: Vec::with_capacity(2 * set_compression as usize),
            cumulative: Vec::<f64>::new(),
            processed_weight: 0.0,
            unprocessed_weight: 0.0,
            min: ::std::f64::MAX,
            max: -::std::f64::MAX,
            variance: 0.0,
            count: 0.0,
            sum: 0.0,
        }
    }

    fn process(&mut self) {
        if self.unprocessed.len() > 0 || self.processed.len() > self.max_processed {
            // Append all processed centroids to the unprocessed list and sort by mean
            self.unprocessed.append(&mut self.processed);

            self.unprocessed
                .sort_unstable_by_key(|a| (a.sort_key1, a.sort_key2));

            // Reset processed list with first unprocessed centroid
            self.processed.clear();
            self.processed.push(self.unprocessed[0].clone());

            self.processed_weight += self.unprocessed_weight;
            self.unprocessed_weight = 0.0;
            let mut so_far: f64 = self.unprocessed[0].weight;
            let mut limit: f64 = self.processed_weight * self.integrated_q(1.0);
            let mut projected: f64;
            let mut k1: f64;
            let mut idx: usize;
            let mut rec: i32 = 0;
            for centroid in self.unprocessed.iter() {
                if rec == 0 {
                    // skip the first unprocessed centroid, emulating range [1:]
                    rec += 1;
                    continue;
                }
                projected = so_far + centroid.weight;
                if projected <= limit {
                    so_far = projected;
                    idx = self.processed.len() - 1;
                    self.processed[idx].add(centroid);
                } else {
                    k1 = self.integrated_location(so_far / self.processed_weight);
                    limit = self.processed_weight * self.integrated_q(k1 + 1.0);
                    so_far += centroid.weight;

                    self.processed.push(centroid::Centroid {
                        mean: centroid.mean,
                        weight: centroid.weight,
                        sort_key1: centroid.mean.floor() as isize,
                        sort_key2: (centroid.mean.signum() * centroid.mean.fract()) as isize,
                    });
                }
            }

            self.update_cumulative();
            self.unprocessed.clear();
        }
    }

    pub fn add_centroid(&mut self, c: centroid::Centroid) {
        self.unprocessed.push(centroid::Centroid {
            mean: c.mean,
            weight: c.weight,
            sort_key1: c.mean.floor() as isize,
            sort_key2: (c.mean.signum() * c.mean.fract()) as isize,
        });
        self.unprocessed_weight += c.weight;

        if self.processed.len() > self.max_processed
            || self.unprocessed.len() > self.max_unprocessed
        {
            self.process();
        }
    }

    pub fn add(&mut self, x: f64, w: f64) {
        if !x.is_nan() {
            // Weighted incremental variance calculation
            //foreach x in the data:
            //  if n=0 then
            //     n = 1
            //     mean = x
            //     S = 0
            //     sumweight = weight
            //  else
            //     n = n + 1
            //     temp = weight + sumweight
            //     S = S + sumweight*weight*(x-mean)^2 / temp
            //     mean = mean + (x-mean)*weight / temp
            //     sumweight = temp
            //  end if
            //end for
            //Variance = S * n / ((n-1) * sumweight)  // if sample is the population, omit n/(n-1)

            self.min = self.min.min(x);
            self.max = self.max.max(x);
            self.sum += x * w;
            self.count += w;
            self.add_centroid(centroid::Centroid {
                mean: x,
                weight: w,
                sort_key1: x.floor() as isize,
                sort_key2: (x.signum() * x.fract()) as isize,
            });
        }
    }

    // adjust all weights by the provided scale factor
    pub fn scale_weight(&mut self, w: f64) {
        if w.is_finite() {
            self.sum *= w;
            self.count *= w;
            self.processed_weight *= w;
            self.unprocessed_weight *= w;

            for p in self.processed.iter_mut() {
                p.weight *= w;
            }

            for p in self.unprocessed.iter_mut() {
                p.weight *= w;
            }
        }
    }

    fn update_cumulative(&mut self) {
        self.cumulative = Vec::<f64>::new();
        let mut prev: f64 = 0.0;
        let mut cur: f64;
        for centroid in self.processed.iter() {
            cur = centroid.weight;
            self.cumulative.push(prev + cur / 2.0);
            prev += cur;
        }
        self.cumulative.push(prev);
    }

    fn integrated_q(&self, k: f64) -> f64 {
        ((k.min(self.compression) * ::std::f64::consts::PI / self.compression
            - ::std::f64::consts::PI / 2.0)
            .sin()
            + 1.0)
            / 2.0
    }

    fn integrated_location(&self, q: f64) -> f64 {
        self.compression * ((2.0 * q - 1.0).asin() + ::std::f64::consts::PI / 2.0)
            / ::std::f64::consts::PI
    }

    pub fn cdf(&mut self, x: f64) -> f64 {
        let width: f64;
        self.process();
        match self.processed.len() {
            0 => return 0.0,
            1 => {
                width = self.max - self.min;
                if x <= self.min {
                    return 0.0;
                }
                if x >= self.max {
                    return 1.0;
                }
                if (x - self.min) <= width {
                    // min and max are too close together to do any viable interpolation
                    return 0.5;
                }
                return (x - self.min) / width;
            }
            _ => (),
        }

        if x <= self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        let m0: f64 = self.processed[0].mean;
        // Left Tail
        if x <= m0 {
            if m0 - self.min > 0.0 {
                return (x - self.min) / (m0 - self.min) * self.processed[0].weight
                    / self.processed_weight
                    / 2.0;
            }
            return 0.0;
        }
        // Right Tail
        let mn: f64 = self.processed[self.processed.len() - 1].mean;
        if x >= mn {
            if self.max - mn > 0.0 {
                return 1.0
                    - (self.max - x) / (self.max - mn)
                        * self.processed[self.processed.len() - 1].weight
                        / self.processed_weight
                        / 2.0;
            }
            return 1.0;
        }

        // Search the processed vector for the first Centroid with (mean > x)
        let mut idx: i32 = -1;
        for centroid in self.processed.iter() {
            idx += 1;
            if centroid.mean > x {
                break;
            }
        }

        let upper: usize = idx as usize;
        let z1: f64 = x - self.processed[upper - 1].mean;
        let z2: f64 = self.processed[upper].mean - x;
        weighted_average(self.cumulative[upper - 1], z2, self.cumulative[upper], z1)
            / self.processed_weight
    }

    pub fn quantile(&mut self, q: f64) -> f64 {
        self.process();
        if q < 0.0 || q > 1.0 || self.processed.len() == 0 {
            return ::std::f64::NAN;
        }
        if self.processed.len() == 1 {
            return self.processed[0].mean;
        }

        let index: f64 = q * self.processed_weight;
        if index < self.processed[0].weight / 2.0 {
            return self.min
                + 2.0 * index / self.processed[0].weight * (self.processed[0].mean - self.min);
        }

        let mut lower: usize = self.cumulative.len() - 1;
        for idx in 0..self.cumulative.len() {
            if self.cumulative[idx] >= index {
                lower = idx as usize;
                break;
            }
        }
        if lower == 0 {
            lower = 1;
        }

        if lower + 1 != self.cumulative.len() && lower > 0 {
            let z1: f64 = index - self.cumulative[lower - 1];
            let z2: f64 = self.cumulative[lower] - index;
            return weighted_average(
                self.processed[lower - 1].mean,
                z2,
                self.processed[lower].mean,
                z1,
            );
        }

        let z1: f64 = index - self.processed_weight - self.processed[lower - 1].weight / 2.0;
        let z2: f64 = (self.processed[lower - 1].weight / 2.0) - z1;
        weighted_average(
            self.processed[self.processed.len() - 1].mean,
            z1,
            self.max,
            z2,
        )
    }

    pub fn print_digest(&mut self) -> String {
        self.process();
        let mut result = format!(
             "{{~variance~:~{}~,~sum~:~{}~,~count~:~{}~,~min~:~{}~,~max~:~{}~,~compression~:~{}~,~max_processed~:~{}~,~max_unprocessed~:~{}~,~processed_weight~:~{}~,~unprocessed_weight~:~{}~,~centroids~: [",
             self.variance(),
             self.sum(),
             self.count(),
             self.min(),
             self.max(),
             self.compression(),
             self.max_processed(),
             self.max_unprocessed(),
             self.processed_weight(),
             self.unprocessed_weight()
         );
        let mut rec: i32 = 0;
        for centroid in self.processed.iter() {
            if rec == 0 {
                rec = 1;
                result = result + &centroid.to_string();
            } else {
                result = result + &",".to_string() + &centroid.to_string();
            }
        }
        result = result + &"]}".to_string();
        result.replace("~", "\x22")
    }

    pub fn save_digest(&mut self, fspec: String) -> std::io::Result<()> {
        ::std::fs::write(fspec, self.print_digest())
    }

    pub fn merge_digest(&mut self, mut td: Tdigest) {
        // Calculate sum, min and max
        let calcsum: f64 = self.sum + td.sum();
        self.min = self.min.min(td.min());
        self.max = self.max.max(td.max());

        // Save some key variables for calculating the merged group variance at end
        // https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
        let n1: f64 = self.count();
        let n2: f64 = td.count();
        let s1_sq: f64 = self.variance();
        let s2_sq: f64 = td.variance();
        let x1: f64 = self.mean();
        let x2: f64 = td.mean();
        let x: f64 = (n1 * x1 + n2 * x2) / (n1 + n2);
        let calcvar: f64 =
            (n1 * s1_sq + n2 * s2_sq + n1 * (x1 - x) * (x1 - x) + n2 * (x2 - x) * (x2 - x))
                / (n1 + n2);

        // We want to add the centroids from the second digest into this one,
        // but not in sorted order as this can be problematic for digests.  Instead,
        // work from the outer edges to the middle of the array until finished
        // [0], [n-1], [1], [n-2], ...
        // Gather these into arrays containing means and weights
        let mut meanvec: Vec<f64> = Vec::new();
        let mut weightvec: Vec<f64> = Vec::new();
        let urange: u32 = td.num_centroids() as u32;
        let ubound: u32 = (f64::from(urange) / 2.0).ceil() as u32;
        let mut c: centroid::Centroid;
        for pos in 0..ubound {
            if pos == urange - pos - 1 {
                c = self.get_centroid_by_index(pos as usize);
                meanvec.push(c.mean);
                weightvec.push(c.weight);
            } else {
                c = self.get_centroid_by_index(pos as usize);
                meanvec.push(c.mean);
                weightvec.push(c.weight);
                c = self.get_centroid_by_index((urange - pos - 1) as usize);
                meanvec.push(c.mean);
                weightvec.push(c.weight);
            }
        }

        // Now rip through the arrays and add them to this digest in the
        // order we harvested them in
        for pos in 0..meanvec.len() {
            self.add(meanvec[pos], weightvec[pos]);
        }
        self.process();
        self.sum = calcsum;
        self.variance = calcvar;
    }

    pub fn load_digest(&mut self, fspec: String) {
        let saveset = read_file_into_string(fspec);
        let digest: Value = serde_json::from_str(&saveset).unwrap();

        // Populate the digest metrics
        self.sum = digest["sum"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.count = digest["count"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.variance = digest["variance"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.min = digest["min"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.max = digest["max"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.compression = digest["compression"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.max_processed = digest["max_processed"]
            .to_string()
            .replace('"', "")
            .parse::<usize>()
            .unwrap();
        self.max_unprocessed = digest["max_unprocessed"]
            .to_string()
            .replace('"', "")
            .parse::<usize>()
            .unwrap();
        self.processed_weight = digest["processed_weight"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();
        self.unprocessed_weight = digest["unprocessed_weight"]
            .to_string()
            .replace('"', "")
            .parse::<f64>()
            .unwrap();

        // Clear out the centroid vectors in this digest and reload
        self.processed.clear();
        self.unprocessed.clear();
        let centroids = json!(digest["centroids"]);
        for pos in 0..(centroids.as_array().unwrap().len()) {
            let centroid = json!(*centroids.get(pos).unwrap());
            let x = centroid["mean"]
                .to_string()
                .replace('"', "")
                .parse::<f64>()
                .unwrap();
            let w = centroid["weight"]
                .to_string()
                .replace('"', "")
                .parse::<f64>()
                .unwrap();
            self.processed.push(centroid::Centroid {
                mean: x,
                weight: w,
                sort_key1: x.floor() as isize,
                sort_key2: (x.signum() * x.fract()) as isize,
            });
        }
        self.update_cumulative();
    }

    pub fn get_centroid_by_index(&mut self, index: usize) -> centroid::Centroid {
        if index < self.num_centroids() {
            centroid::Centroid {
                mean: self.processed[index].mean,
                weight: self.processed[index].weight,
                sort_key1: self.processed[index].sort_key1,
                sort_key2: self.processed[index].sort_key2,
            }
        } else {
            centroid::Centroid {
                mean: 0.0,
                weight: 0.0,
                sort_key1: 0,
                sort_key2: 0,
            }
        }
    }

    pub fn num_centroids(&mut self) -> usize {
        self.processed.len()
    }

    pub fn count(&mut self) -> f64 {
        self.process();
        self.count
    }

    pub fn mean(&self) -> f64 {
        self.sum / self.count
    }

    pub fn variance(&self) -> f64 {
        self.variance
    }

    pub fn stdev(&self) -> f64 {
        self.variance.sqrt()
    }

    pub fn sum(&mut self) -> f64 {
        self.sum
    }

    pub fn min(&mut self) -> f64 {
        self.min
    }

    pub fn max(&mut self) -> f64 {
        self.max
    }

    pub fn compression(&mut self) -> f64 {
        self.compression
    }

    pub fn max_processed(&mut self) -> usize {
        self.max_processed
    }

    pub fn max_unprocessed(&mut self) -> usize {
        self.max_unprocessed
    }

    pub fn processed_weight(&mut self) -> f64 {
        self.processed_weight
    }

    pub fn unprocessed_weight(&mut self) -> f64 {
        self.unprocessed_weight
    }

    pub fn total_weight(&self) -> f64 {
        self.processed_weight + self.unprocessed_weight
    }
}

fn weighted_average(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    if x1 <= x2 {
        weighted_average_sorted(x1, w1, x2, w2);
    }
    weighted_average_sorted(x2, w2, x1, w1)
}

fn weighted_average_sorted(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    x1.max(x2.min((x1 * w1 + x2 * w2) / (w1 + w2)))
}

pub fn apply1(var: &Tdigest, op: impl Fn(f64) -> f64) -> Tdigest {
    let mut new_var = Tdigest::new(var.compression);

    for c in var.processed.iter().chain(var.unprocessed.iter()) {
        new_var.add(op(c.mean), c.weight)
    }

    new_var
}

pub fn apply2(var1: &Tdigest, var2: &Tdigest, op: impl Fn(f64, f64) -> f64) -> Tdigest {
    let mut new_var = Tdigest::new(var1.compression.min(var2.compression));

    for c1 in var1.processed.iter().chain(var1.unprocessed.iter()) {
        for c2 in var2.processed.iter().chain(var2.unprocessed.iter()) {
            new_var.add(op(c1.mean, c2.mean), c1.weight * c2.weight)
        }
    }

    new_var
}

pub fn read_file_into_string(fspec: String) -> String {
    let mut contents: String = "".to_string();
    let f = File::open(&fspec);
    let f = match f {
        Ok(file) => file,
        Err(e) => {
            panic!("Failed to open dataset {}: {}", fspec, e);
        }
    };
    for line in BufReader::new(f).lines() {
        match line {
            Err(e) => {
                panic!("Failed reading data line in {}: {}", fspec, e);
            }
            Ok(str) => {
                contents = format!("{}{}", contents, str);
            }
        }
    }
    contents
}

fn processed_size(size: usize, compression: f64) -> usize {
    if size == 0 {
        return (2.0 * compression.ceil()) as usize;
    }
    size
}

fn unprocessed_size(size: usize, compression: f64) -> usize {
    if size == 0 {
        return (8.0 * compression.ceil()).floor() as usize;
    }
    size
}
