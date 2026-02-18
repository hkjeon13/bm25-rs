use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rayon::prelude::*;

use crate::BM25;

/// Min-heap wrapper for top-N selection — only holds score + doc_id (no text)
pub(crate) struct MinScore(pub f32, pub String);

impl PartialEq for MinScore {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for MinScore {}

impl PartialOrd for MinScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order so BinaryHeap acts as min-heap
        // Use total_cmp for deterministic NaN handling
        other.0.total_cmp(&self.0)
    }
}

impl BM25 {
    pub(crate) fn calculate_score(
        tf: f32,
        df: f32,
        doc_len: usize,
        num_docs: f32,
        average_length: f32,
        k1: f32,
        b: f32,
    ) -> f32 {
        if average_length <= 0.0 || num_docs <= 0.0 {
            return 0.0;
        }
        (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len as f32 / average_length)))
            * (((num_docs + 1.0) / (df + 1.0)).ln() + 1.0)
    }

    pub(crate) fn update_freeze_map(&mut self) {
        if self.doc_len_map.is_empty() {
            self.average_length = 0.0;
            self.freeze_map.clear();
            return;
        }

        self.refresh_average_length();
        let num_docs = self.doc_len_map.len() as f32;

        let entries: Vec<_> = self.index_map.iter().collect();
        self.freeze_map = entries
            .par_iter()
            .map(|(token, doc_freq)| {
                (
                    (*token).clone(),
                    doc_freq
                        .iter()
                        .map(|(doc_id, &tf)| {
                            let doc_len = *self.doc_len_map.get(doc_id).unwrap_or(&0);
                            let df = doc_freq.len() as f32;
                            let score = Self::calculate_score(
                                tf as f32,
                                df,
                                doc_len,
                                num_docs,
                                self.average_length,
                                self.k1,
                                self.b,
                            );
                            (doc_id.clone(), score)
                        })
                        .collect(),
                )
            })
            .collect();
    }

    pub(crate) fn ensure_frozen(&self) -> Result<(), &'static str> {
        if !self.is_frozen {
            return Err("Index is not frozen. Call freeze() before search().");
        }
        Ok(())
    }

    pub(crate) fn deduplicate_tokens(tokens: &[String]) -> Vec<&str> {
        let mut seen = HashSet::with_capacity(tokens.len());
        tokens
            .iter()
            .filter(|t| seen.insert(t.as_str()))
            .map(|t| t.as_str())
            .collect()
    }

    // BinaryHeap top-N selection — O(D log N) instead of O(D log D)
    // Text is fetched only for the final N results, not during heap operations
    pub(crate) fn collect_results(
        &self,
        scores: HashMap<&str, f32>,
        n: usize,
    ) -> Vec<(f32, String, String)> {
        let mut heap: BinaryHeap<MinScore> = BinaryHeap::with_capacity(n + 1);

        for (&id, &score) in &scores {
            heap.push(MinScore(score, id.to_owned()));
            if heap.len() > n {
                heap.pop();
            }
        }

        let mut results: Vec<_> = heap
            .into_iter()
            .map(|ms| {
                let text = self.doc_texts.get(&ms.1).cloned().unwrap_or_default();
                (ms.0, ms.1, text)
            })
            .collect();
        results.sort_by(|a, b| b.0.total_cmp(&a.0));
        results
    }

    pub(crate) fn search_frozen_internal(
        &self,
        query_tokens: &[String],
        n: usize,
    ) -> Vec<(f32, String, String)> {
        let unique_tokens = Self::deduplicate_tokens(query_tokens);

        let estimated = unique_tokens
            .iter()
            .filter_map(|t| self.freeze_map.get(*t))
            .map(|m| m.len())
            .sum::<usize>()
            .min(self.doc_len_map.len());

        let mut scores: HashMap<&str, f32> = HashMap::with_capacity(estimated);

        for token in &unique_tokens {
            if let Some(doc_scores) = self.freeze_map.get(*token) {
                for (doc_id, &score) in doc_scores {
                    *scores.entry(doc_id.as_str()).or_insert(0.0) += score;
                }
            }
        }

        self.collect_results(scores, n)
    }

    pub(crate) fn search_instance_internal(
        &self,
        query_tokens: &[String],
        n: usize,
    ) -> Vec<(f32, String, String)> {
        if self.doc_len_map.is_empty() {
            return Vec::new();
        }

        let unique_tokens = Self::deduplicate_tokens(query_tokens);

        let num_docs = self.doc_len_map.len() as f32;
        let average_length = self.average_length;

        let estimated = unique_tokens
            .iter()
            .filter_map(|t| self.index_map.get(*t))
            .map(|m| m.len())
            .sum::<usize>()
            .min(self.doc_len_map.len());

        let mut scores: HashMap<&str, f32> = HashMap::with_capacity(estimated);

        for token in &unique_tokens {
            if let Some(doc_freq) = self.index_map.get(*token) {
                let df = doc_freq.len() as f32;
                for (doc_id, &tf) in doc_freq {
                    let doc_len = *self.doc_len_map.get(doc_id).unwrap_or(&0);
                    let score = Self::calculate_score(
                        tf as f32,
                        df,
                        doc_len,
                        num_docs,
                        average_length,
                        self.k1,
                        self.b,
                    );
                    *scores.entry(doc_id.as_str()).or_insert(0.0) += score;
                }
            }
        }

        self.collect_results(scores, n)
    }
}
