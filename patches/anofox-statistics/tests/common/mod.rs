use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

/// Data for two-way ANOVA tests: values with parallel factor level arrays.
pub struct TwoWayData {
    pub values: Vec<f64>,
    pub factor_a: Vec<usize>,
    pub factor_b: Vec<usize>,
}

/// Load two-way ANOVA data from a CSV file with columns: value, factor_a, factor_b.
pub fn load_two_way_data(filename: &str) -> TwoWayData {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data");
    path.push(filename);

    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "Could not open reference file '{}': {}. Did you run 'Rscript R/generate_refs.R'?",
            path.display(),
            e
        )
    });

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut values = Vec::new();
    let mut factor_a = Vec::new();
    let mut factor_b = Vec::new();

    for record_result in rdr.records() {
        let record = record_result.unwrap();
        values.push(record.get(0).unwrap().parse::<f64>().unwrap());
        factor_a.push(record.get(1).unwrap().parse::<usize>().unwrap());
        factor_b.push(record.get(2).unwrap().parse::<usize>().unwrap());
    }

    TwoWayData {
        values,
        factor_a,
        factor_b,
    }
}

/// Load a single-row CSV of scalar reference values into a HashMap.
/// Each column becomes a key, its value becomes the f64 value.
pub fn load_reference_scalars(filename: &str) -> HashMap<String, f64> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data");
    path.push(filename);

    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "Could not open reference file '{}': {}. Did you run 'Rscript R/generate_refs.R'?",
            path.display(),
            e
        )
    });

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let headers = rdr.headers().unwrap().clone();
    let mut result = HashMap::new();

    if let Some(record_result) = rdr.records().next() {
        let record = record_result.unwrap();
        for (i, field) in record.iter().enumerate() {
            let value: f64 = field
                .parse()
                .unwrap_or_else(|e| panic!("Could not parse '{}' as f64: {}", field, e));
            result.insert(headers[i].to_string(), value);
        }
    }

    result
}

/// Data for repeated measures ANOVA tests: matrix format (rows = subjects, cols = conditions).
pub struct RmAnovaData {
    pub subjects: Vec<Vec<f64>>, // Each inner Vec is one subject's data across conditions
    pub n_subjects: usize,
    pub n_conditions: usize,
}

/// Load repeated measures ANOVA data from a CSV file with columns: subject, condition, value.
/// Returns data in matrix format suitable for the repeated_measures_anova function.
pub fn load_rm_data(filename: &str) -> RmAnovaData {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data");
    path.push(filename);

    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "Could not open reference file '{}': {}. Did you run 'Rscript R/generate_refs.R'?",
            path.display(),
            e
        )
    });

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    // Collect all rows and find max subject/condition indices
    let mut rows: Vec<(usize, usize, f64)> = Vec::new();
    let mut max_subject = 0;
    let mut max_condition = 0;

    for record_result in rdr.records() {
        let record = record_result.unwrap();
        let subject: usize = record.get(0).unwrap().parse().unwrap();
        let condition: usize = record.get(1).unwrap().parse().unwrap();
        let value: f64 = record.get(2).unwrap().parse().unwrap();

        max_subject = max_subject.max(subject);
        max_condition = max_condition.max(condition);
        rows.push((subject, condition, value));
    }

    let n_subjects = max_subject; // 1-indexed in the file
    let n_conditions = max_condition + 1; // 0-indexed in the file

    // Initialize matrix
    let mut subjects: Vec<Vec<f64>> = vec![vec![0.0; n_conditions]; n_subjects];

    // Fill in values
    for (subject, condition, value) in rows {
        subjects[subject - 1][condition] = value;
    }

    RmAnovaData {
        subjects,
        n_subjects,
        n_conditions,
    }
}

/// Load a single-column CSV of vector values.
pub fn load_reference_vector(filename: &str) -> Vec<f64> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data");
    path.push(filename);

    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "Could not open reference file '{}': {}. Did you run 'Rscript R/generate_refs.R'?",
            path.display(),
            e
        )
    });

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut result = Vec::new();

    for record_result in rdr.records() {
        let record = record_result.unwrap();
        if let Some(field) = record.get(0) {
            let value: f64 = field
                .parse()
                .unwrap_or_else(|e| panic!("Could not parse '{}' as f64: {}", field, e));
            result.push(value);
        }
    }

    result
}
