use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::{BufReader, BufWriter};

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Min-heap wrapper for top-N selection — only holds score + doc_id (no text)
struct MinScore(f32, String);

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

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
struct BM25 {
    index_map: HashMap<String, HashMap<String, u32>>,
    doc_len_map: HashMap<String, usize>,
    doc_texts: HashMap<String, String>,
    #[serde(skip)]
    freeze_map: HashMap<String, HashMap<String, f32>>,
    k1: f32,
    b: f32,
    average_length: f32,
    #[serde(default)]
    is_frozen: bool,
}

impl BM25 {
    fn calculate_score(
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

    fn update_average_length(&mut self) {
        if self.doc_len_map.is_empty() {
            self.average_length = 0.0;
        } else {
            let num_docs = self.doc_len_map.len() as f32;
            self.average_length = self.doc_len_map.values().sum::<usize>() as f32 / num_docs;
        }
    }

    fn update_freeze_map(&mut self) {
        if self.doc_len_map.is_empty() {
            self.average_length = 0.0;
            self.freeze_map.clear();
            return;
        }

        self.update_average_length();
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

    fn mark_unfrozen(&mut self) {
        self.is_frozen = false;
        self.freeze_map.clear();
    }

    fn remove_document_from_index(&mut self, id: &str) {
        self.index_map.retain(|_, posting| {
            posting.remove(id);
            !posting.is_empty()
        });
    }

    fn ensure_frozen(&self) -> PyResult<()> {
        if !self.is_frozen {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "Index is not frozen. Call freeze() before search().",
            ));
        }
        Ok(())
    }

    // BinaryHeap top-N selection — O(D log N) instead of O(D log D)
    // Text is fetched only for the final N results, not during heap operations
    fn collect_results(
        &self,
        scores: HashMap<&str, f32>,
        n: usize,
    ) -> Vec<(f32, String, String)> {
        let mut heap: BinaryHeap<MinScore> = BinaryHeap::with_capacity(n + 1);

        for (&id, &score) in &scores {
            heap.push(MinScore(score, id.to_owned()));
            if heap.len() > n {
                heap.pop(); // Remove smallest
            }
        }

        let mut results: Vec<_> = heap
            .into_iter()
            .map(|ms| {
                let text = self.doc_texts.get(&ms.1).cloned().unwrap_or_default();
                (ms.0, ms.1, text)
            })
            .collect();
        // Sort descending by score
        results.sort_by(|a, b| b.0.total_cmp(&a.0));
        results
    }

    fn search_frozen_internal(
        &self,
        query_tokens: &[String],
        n: usize,
    ) -> Vec<(f32, String, String)> {
        let estimated = query_tokens
            .iter()
            .filter_map(|t| self.freeze_map.get(t))
            .map(|m| m.len())
            .sum::<usize>()
            .min(self.doc_len_map.len());

        let mut scores: HashMap<&str, f32> = HashMap::with_capacity(estimated);

        for token in query_tokens {
            if let Some(doc_scores) = self.freeze_map.get(token) {
                for (doc_id, &score) in doc_scores {
                    *scores.entry(doc_id.as_str()).or_insert(0.0) += score;
                }
            }
        }

        self.collect_results(scores, n)
    }

    fn search_instance_internal(
        &self,
        query_tokens: &[String],
        n: usize,
    ) -> Vec<(f32, String, String)> {
        if self.doc_len_map.is_empty() {
            return Vec::new();
        }

        let num_docs = self.doc_len_map.len() as f32;
        // Use the cached average_length instead of recalculating
        let average_length = self.average_length;

        let estimated = query_tokens
            .iter()
            .filter_map(|t| self.index_map.get(t))
            .map(|m| m.len())
            .sum::<usize>()
            .min(self.doc_len_map.len());

        let mut scores: HashMap<&str, f32> = HashMap::with_capacity(estimated);

        for token in query_tokens {
            if let Some(doc_freq) = self.index_map.get(token) {
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

#[pymethods]
impl BM25 {
    #[new]
    fn new() -> Self {
        BM25 {
            index_map: HashMap::new(),
            doc_len_map: HashMap::new(),
            doc_texts: HashMap::new(),
            freeze_map: HashMap::new(),
            k1: 1.5,
            b: 0.75,
            average_length: 0.0,
            is_frozen: false,
        }
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        py.allow_threads(|| {
            let file = std::fs::File::open(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to read file: {e}")))?;
            let reader = BufReader::new(file);
            let mut loaded: Self = serde_json::from_reader(reader)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid BM25 JSON: {e}")))?;
            loaded.update_average_length();
            loaded.is_frozen = false;
            Ok(loaded)
        })
    }

    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let file = std::fs::File::create(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to write file: {e}")))?;
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &self)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Unable to serialize BM25: {e}")))?;
            Ok(())
        })
    }

    #[classmethod]
    fn load_bin(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        py.allow_threads(|| {
            let file = std::fs::File::open(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to read file: {e}")))?;
            let reader = BufReader::new(file);
            let mut loaded: Self = bincode::deserialize_from(reader)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid BM25 binary: {e}")))?;
            loaded.update_average_length();
            loaded.is_frozen = false;
            Ok(loaded)
        })
    }

    fn save_bin(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let file = std::fs::File::create(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to write file: {e}")))?;
            let writer = BufWriter::new(file);
            bincode::serialize_into(writer, &self)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Unable to serialize BM25: {e}")))?;
            Ok(())
        })
    }

    #[classmethod]
    fn load_msgpack(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        py.allow_threads(|| {
            let file = std::fs::File::open(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to read file: {e}")))?;
            let reader = BufReader::new(file);
            let mut loaded: Self = rmp_serde::from_read(reader)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid BM25 msgpack: {e}")))?;
            loaded.update_average_length();
            loaded.is_frozen = false;
            Ok(loaded)
        })
    }

    fn save_msgpack(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let file = std::fs::File::create(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to write file: {e}")))?;
            let mut writer = BufWriter::new(file);
            rmp_serde::encode::write(&mut writer, &self)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Unable to serialize BM25: {e}")))?;
            Ok(())
        })
    }

    fn get_freeze_map(&self) -> PyResult<HashMap<String, HashMap<String, f32>>> {
        Ok(self.freeze_map.clone())
    }

    fn set_k1(&mut self, k1: f32) -> PyResult<()> {
        if k1 <= 0.0 {
            return Err(PyErr::new::<PyValueError, _>("k1 must be > 0."));
        }
        self.k1 = k1;
        self.mark_unfrozen();
        Ok(())
    }

    fn set_b(&mut self, b: f32) -> PyResult<()> {
        if !(0.0..=1.0).contains(&b) {
            return Err(PyErr::new::<PyValueError, _>("b must be in [0, 1]."));
        }
        self.b = b;
        self.mark_unfrozen();
        Ok(())
    }

    fn get_doc_texts(&self) -> PyResult<HashMap<String, String>> {
        Ok(self.doc_texts.clone())
    }

    fn get_index_map(&self) -> PyResult<HashMap<String, HashMap<String, u32>>> {
        Ok(self.index_map.clone())
    }

    fn get_doc_len_map(&self) -> PyResult<HashMap<String, usize>> {
        Ok(self.doc_len_map.clone())
    }

    fn set_doc_texts(&mut self, doc_texts: HashMap<String, String>) {
        self.doc_texts = doc_texts;
    }

    fn set_index_map(&mut self, index_map: HashMap<String, HashMap<String, u32>>) {
        self.index_map = index_map;
        self.mark_unfrozen();
    }

    fn set_doc_len_map(&mut self, doc_len_map: HashMap<String, usize>) {
        self.doc_len_map = doc_len_map;
        self.update_average_length();
        self.mark_unfrozen();
    }

    fn add_document(&mut self, id: String, tokens: Vec<String>, text: String) -> PyResult<()> {
        if id.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("Document id must not be empty."));
        }
        if tokens.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("Tokens must not be empty."));
        }

        self.remove_document_from_index(&id);

        for token in tokens.iter() {
            let target = self.index_map.entry(token.to_string()).or_default();
            *target.entry(id.clone()).or_insert(0) += 1;
        }
        self.doc_len_map.insert(id.clone(), tokens.len());
        self.doc_texts.insert(id, text);
        self.update_average_length();
        self.mark_unfrozen();
        Ok(())
    }

    fn add_documents(&mut self, documents: Vec<(String, Vec<String>, String)>) -> PyResult<()> {
        for (id, tokens, _) in &documents {
            if id.is_empty() {
                return Err(PyErr::new::<PyValueError, _>("Document id must not be empty."));
            }
            if tokens.is_empty() {
                return Err(PyErr::new::<PyValueError, _>(
                    format!("Tokens must not be empty for document '{}'.", id),
                ));
            }
        }

        for (id, tokens, text) in documents {
            self.remove_document_from_index(&id);

            for token in tokens.iter() {
                let target = self.index_map.entry(token.to_string()).or_default();
                *target.entry(id.clone()).or_insert(0) += 1;
            }
            self.doc_len_map.insert(id.clone(), tokens.len());
            self.doc_texts.insert(id, text);
        }

        self.update_average_length();
        // Only mark unfrozen once after all documents are added
        self.mark_unfrozen();
        Ok(())
    }

    fn freeze(&mut self, py: Python<'_>) {
        py.allow_threads(|| {
            self.update_freeze_map();
            self.is_frozen = true;
        });
    }

    fn search(&self, query_tokens: Vec<String>, n: usize) -> PyResult<Vec<(f32, String, String)>> {
        if n == 0 {
            return Err(PyErr::new::<PyValueError, _>("n must be > 0."));
        }
        self.ensure_frozen()?;
        Ok(self.search_frozen_internal(&query_tokens, n))
    }

    fn search_instance(
        &self,
        query_tokens: Vec<String>,
        n: usize,
    ) -> PyResult<Vec<(f32, String, String)>> {
        if n == 0 {
            return Err(PyErr::new::<PyValueError, _>("n must be > 0."));
        }
        Ok(self.search_instance_internal(&query_tokens, n))
    }

    fn batch_search(
        &self,
        py: Python<'_>,
        tokenized_queries: Vec<Vec<String>>,
        n: usize,
    ) -> PyResult<Vec<Vec<(f32, String, String)>>> {
        if n == 0 {
            return Err(PyErr::new::<PyValueError, _>("n must be > 0."));
        }
        self.ensure_frozen()?;
        Ok(py.allow_threads(|| {
            tokenized_queries
                .par_iter()
                .map(|tokenized_query| self.search_frozen_internal(tokenized_query, n))
                .collect()
        }))
    }

    fn batch_search_instance(
        &self,
        py: Python<'_>,
        tokenized_queries: Vec<Vec<String>>,
        n: usize,
    ) -> PyResult<Vec<Vec<(f32, String, String)>>> {
        if n == 0 {
            return Err(PyErr::new::<PyValueError, _>("n must be > 0."));
        }
        Ok(py.allow_threads(|| {
            tokenized_queries
                .par_iter()
                .map(|tokenized_query| self.search_instance_internal(tokenized_query, n))
                .collect()
        }))
    }

    fn remove_document(&mut self, id: String) -> PyResult<()> {
        if !self.doc_len_map.contains_key(&id) {
            return Err(PyErr::new::<PyValueError, _>(
                format!("Document '{}' does not exist.", id),
            ));
        }
        self.remove_document_from_index(&id);
        self.doc_len_map.remove(&id);
        self.doc_texts.remove(&id);
        self.update_average_length();
        self.mark_unfrozen();
        Ok(())
    }

    fn doc_count(&self) -> usize {
        self.doc_len_map.len()
    }

    fn contains_document(&self, id: String) -> bool {
        self.doc_len_map.contains_key(&id)
    }
}

#[pymodule]
fn bm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bm25() -> BM25 {
        BM25 {
            index_map: HashMap::new(),
            doc_len_map: HashMap::new(),
            doc_texts: HashMap::new(),
            freeze_map: HashMap::new(),
            k1: 1.5,
            b: 0.75,
            average_length: 0.0,
            is_frozen: false,
        }
    }

    fn add_doc(bm25: &mut BM25, id: &str, tokens: Vec<&str>, text: &str) {
        let tokens: Vec<String> = tokens.into_iter().map(String::from).collect();
        let id = id.to_string();

        bm25.remove_document_from_index(&id);
        for token in tokens.iter() {
            let target = bm25.index_map.entry(token.to_string()).or_default();
            *target.entry(id.clone()).or_insert(0) += 1;
        }
        bm25.doc_len_map.insert(id.clone(), tokens.len());
        bm25.doc_texts.insert(id, text.to_string());
        bm25.update_average_length();
    }

    #[test]
    fn test_calculate_score_basic() {
        let score = BM25::calculate_score(1.0, 1.0, 10, 100.0, 10.0, 1.5, 0.75);
        assert!(score > 0.0);
        assert!(score.is_finite());
    }

    #[test]
    fn test_calculate_score_zero_average_length() {
        let score = BM25::calculate_score(1.0, 1.0, 10, 100.0, 0.0, 1.5, 0.75);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_calculate_score_zero_num_docs() {
        let score = BM25::calculate_score(1.0, 1.0, 10, 0.0, 10.0, 1.5, 0.75);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_calculate_score_higher_tf_higher_score() {
        let s1 = BM25::calculate_score(1.0, 1.0, 10, 100.0, 10.0, 1.5, 0.75);
        let s2 = BM25::calculate_score(5.0, 1.0, 10, 100.0, 10.0, 1.5, 0.75);
        assert!(s2 > s1, "higher TF should yield higher score");
    }

    #[test]
    fn test_calculate_score_higher_df_lower_score() {
        let s1 = BM25::calculate_score(1.0, 1.0, 10, 100.0, 10.0, 1.5, 0.75);
        let s2 = BM25::calculate_score(1.0, 50.0, 10, 100.0, 10.0, 1.5, 0.75);
        assert!(s1 > s2, "higher DF should yield lower score (less rare)");
    }

    #[test]
    fn test_empty_index_search() {
        let bm25 = make_bm25();
        let query = vec!["hello".to_string()];
        let results = bm25.search_instance_internal(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_add_and_search() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
        add_doc(&mut bm25, "d2", vec!["hello", "rust"], "hello rust");
        add_doc(&mut bm25, "d3", vec!["foo", "bar"], "foo bar");

        let query = vec!["hello".to_string()];
        let results = bm25.search_instance_internal(&query, 5);
        assert_eq!(results.len(), 2);
        // Both d1 and d2 should appear
        let ids: Vec<&str> = results.iter().map(|r| r.1.as_str()).collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d2"));
    }

    #[test]
    fn test_search_top_n_limits_results() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello"], "hello");
        add_doc(&mut bm25, "d2", vec!["hello"], "hello");
        add_doc(&mut bm25, "d3", vec!["hello"], "hello");

        let query = vec!["hello".to_string()];
        let results = bm25.search_instance_internal(&query, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove_document() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
        add_doc(&mut bm25, "d2", vec!["hello", "rust"], "hello rust");

        bm25.remove_document_from_index("d1");
        bm25.doc_len_map.remove("d1");
        bm25.doc_texts.remove("d1");
        bm25.update_average_length();

        assert!(!bm25.doc_len_map.contains_key("d1"));
        // "world" token should be fully removed since d1 was its only doc
        assert!(!bm25.index_map.contains_key("world"));
        // "hello" token should still exist for d2
        assert!(bm25.index_map.contains_key("hello"));
    }

    #[test]
    fn test_duplicate_doc_id_replaces() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
        add_doc(&mut bm25, "d1", vec!["foo", "bar"], "foo bar");

        assert_eq!(bm25.doc_len_map.len(), 1);
        assert_eq!(bm25.doc_texts.get("d1").unwrap(), "foo bar");
        // "hello" and "world" should no longer be in index
        assert!(!bm25.index_map.contains_key("hello"));
        assert!(!bm25.index_map.contains_key("world"));
    }

    #[test]
    fn test_freeze_and_frozen_search() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
        add_doc(&mut bm25, "d2", vec!["hello", "rust"], "hello rust");

        bm25.update_freeze_map();
        bm25.is_frozen = true;

        let query = vec!["hello".to_string()];
        let results = bm25.search_frozen_internal(&query, 5);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_average_length_maintained() {
        let mut bm25 = make_bm25();
        assert_eq!(bm25.average_length, 0.0);

        add_doc(&mut bm25, "d1", vec!["a", "b", "c"], "a b c");
        assert!((bm25.average_length - 3.0).abs() < f32::EPSILON);

        add_doc(&mut bm25, "d2", vec!["x"], "x");
        // (3 + 1) / 2 = 2.0
        assert!((bm25.average_length - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_results_sorted_descending() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello"], "hello");
        add_doc(&mut bm25, "d2", vec!["hello", "hello", "hello"], "hello hello hello");
        add_doc(&mut bm25, "d3", vec!["hello", "hello"], "hello hello");

        let query = vec!["hello".to_string()];
        let results = bm25.search_instance_internal(&query, 10);
        for window in results.windows(2) {
            assert!(window[0].0 >= window[1].0, "results should be sorted descending by score");
        }
    }

    #[test]
    fn test_nan_handling_in_minscore() {
        // NaN scores should be handled deterministically via total_cmp
        let a = MinScore(f32::NAN, "a".to_string());
        let b = MinScore(1.0, "b".to_string());
        // Just ensure no panic — total_cmp handles NaN deterministically
        let _ = a.cmp(&b);
        let _ = b.cmp(&a);
        let _ = a.eq(&b);
    }

    #[test]
    fn test_serde_skip_freeze_map() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello"], "hello");
        bm25.update_freeze_map();
        bm25.is_frozen = true;
        assert!(!bm25.freeze_map.is_empty());

        let json = serde_json::to_string(&bm25).unwrap();
        // freeze_map should not appear in serialized output
        assert!(!json.contains("freeze_map"));

        let loaded: BM25 = serde_json::from_str(&json).unwrap();
        assert!(loaded.freeze_map.is_empty());
    }

    #[test]
    fn test_retain_removes_empty_tokens() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["unique_token", "shared"], "text");
        add_doc(&mut bm25, "d2", vec!["shared"], "text2");

        bm25.remove_document_from_index("d1");
        // "unique_token" should be gone, "shared" should remain
        assert!(!bm25.index_map.contains_key("unique_token"));
        assert!(bm25.index_map.contains_key("shared"));
    }

    #[test]
    fn test_no_text_query_match() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello"], "hello");

        let query = vec!["nonexistent".to_string()];
        let results = bm25.search_instance_internal(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bincode_roundtrip() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
        add_doc(&mut bm25, "d2", vec!["foo", "bar"], "foo bar");

        let encoded = bincode::serialize(&bm25).unwrap();
        let decoded: BM25 = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.doc_len_map.len(), 2);
        assert_eq!(decoded.doc_texts.get("d1").unwrap(), "hello world");
        assert_eq!(decoded.doc_texts.get("d2").unwrap(), "foo bar");
        assert!(decoded.freeze_map.is_empty()); // skipped by serde
    }

    #[test]
    fn test_msgpack_roundtrip() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
        add_doc(&mut bm25, "d2", vec!["foo", "bar"], "foo bar");

        let encoded = rmp_serde::to_vec(&bm25).unwrap();
        let decoded: BM25 = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.doc_len_map.len(), 2);
        assert_eq!(decoded.doc_texts.get("d1").unwrap(), "hello world");
        assert_eq!(decoded.doc_texts.get("d2").unwrap(), "foo bar");
        assert!(decoded.freeze_map.is_empty());
    }

    #[test]
    fn test_json_streaming_roundtrip() {
        let mut bm25 = make_bm25();
        add_doc(&mut bm25, "d1", vec!["hello"], "hello");

        // Simulate streaming write + read
        let mut buf = Vec::new();
        serde_json::to_writer(&mut buf, &bm25).unwrap();
        let loaded: BM25 = serde_json::from_reader(buf.as_slice()).unwrap();

        assert_eq!(loaded.doc_len_map.len(), 1);
        assert_eq!(loaded.doc_texts.get("d1").unwrap(), "hello");
    }
}
