use crate::error::{Result, StatError};

/// Sort data by value and return index-value pairs.
fn sort_indexed(data: &[f64]) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indexed
}

/// Find the end index of a tie group starting at `start`.
fn find_tie_group_end(indexed: &[(usize, f64)], start: usize) -> usize {
    let value = indexed[start].1;
    indexed[start..]
        .iter()
        .take_while(|(_, v)| *v == value)
        .count()
        + start
}

/// Assign average rank to a tie group and optionally record tie size.
fn assign_tie_group_ranks(
    indexed: &[(usize, f64)],
    ranks: &mut [f64],
    start: usize,
    end: usize,
    tie_sizes: Option<&mut Vec<usize>>,
) {
    let avg_rank = (start + 1 + end) as f64 / 2.0;
    for item in indexed.iter().take(end).skip(start) {
        ranks[item.0] = avg_rank;
    }
    if let Some(ties) = tie_sizes {
        let tie_size = end - start;
        if tie_size > 1 {
            ties.push(tie_size);
        }
    }
}

/// Compute ranks of data with average tie handling (matching R's rank(ties.method="average")).
///
/// # Arguments
/// * `data` - The data to rank
///
/// # Returns
/// * Vector of ranks (1-indexed, ties get average rank)
pub fn rank(data: &[f64]) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let indexed = sort_indexed(data);
    let mut ranks = vec![0.0; data.len()];

    let mut i = 0;
    while i < indexed.len() {
        let j = find_tie_group_end(&indexed, i);
        assign_tie_group_ranks(&indexed, &mut ranks, i, j, None);
        i = j;
    }

    Ok(ranks)
}

/// Internal helper: compute ranks and return tie information for correction
pub(crate) fn rank_with_ties(data: &[f64]) -> Result<(Vec<f64>, Vec<usize>)> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let indexed = sort_indexed(data);
    let mut ranks = vec![0.0; data.len()];
    let mut tie_sizes = Vec::new();

    let mut i = 0;
    while i < indexed.len() {
        let j = find_tie_group_end(&indexed, i);
        assign_tie_group_ranks(&indexed, &mut ranks, i, j, Some(&mut tie_sizes));
        i = j;
    }

    Ok((ranks, tie_sizes))
}
