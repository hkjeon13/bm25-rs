pub(crate) mod scoring;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::io::{BufReader, BufWriter};

use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct BM25 {
    pub(crate) index_map: HashMap<String, HashMap<String, u32>>,
    pub(crate) doc_len_map: HashMap<String, usize>,
    pub(crate) doc_texts: HashMap<String, String>,
    #[serde(skip)]
    pub(crate) freeze_map: HashMap<String, HashMap<String, f32>>,
    pub(crate) k1: f32,
    pub(crate) b: f32,
    pub(crate) average_length: f32,
    #[serde(default)]
    pub(crate) total_length: usize,
    #[serde(default)]
    pub(crate) is_frozen: bool,
}

impl BM25 {
    pub(crate) fn recompute_stats(&mut self) {
        if self.doc_len_map.is_empty() {
            self.total_length = 0;
            self.average_length = 0.0;
        } else {
            self.total_length = self.doc_len_map.values().sum();
            self.average_length = self.total_length as f32 / self.doc_len_map.len() as f32;
        }
    }

    pub(crate) fn refresh_average_length(&mut self) {
        if self.doc_len_map.is_empty() {
            self.average_length = 0.0;
        } else {
            self.average_length = self.total_length as f32 / self.doc_len_map.len() as f32;
        }
    }

    pub(crate) fn mark_unfrozen(&mut self) {
        self.is_frozen = false;
        self.freeze_map.clear();
    }

    pub(crate) fn remove_document_from_index(&mut self, id: &str) {
        self.index_map.retain(|_, posting| {
            posting.remove(id);
            !posting.is_empty()
        });
    }
}

#[pymethods]
impl BM25 {
    // --- Constructor & dunder methods ---

    #[new]
    #[pyo3(signature = (k1=1.5, b=0.75))]
    fn new(k1: f32, b: f32) -> PyResult<Self> {
        if k1 <= 0.0 {
            return Err(PyErr::new::<PyValueError, _>("k1 must be > 0."));
        }
        if !(0.0..=1.0).contains(&b) {
            return Err(PyErr::new::<PyValueError, _>("b must be in [0, 1]."));
        }
        Ok(BM25 {
            index_map: HashMap::new(),
            doc_len_map: HashMap::new(),
            doc_texts: HashMap::new(),
            freeze_map: HashMap::new(),
            k1,
            b,
            average_length: 0.0,
            total_length: 0,
            is_frozen: false,
        })
    }

    fn __len__(&self) -> usize {
        self.doc_len_map.len()
    }

    fn __contains__(&self, id: &str) -> bool {
        self.doc_len_map.contains_key(id)
    }

    fn __repr__(&self) -> String {
        format!(
            "BM25(docs={}, k1={}, b={}, frozen={})",
            self.doc_len_map.len(),
            self.k1,
            self.b,
            self.is_frozen,
        )
    }

    // --- Persistence ---

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        py.allow_threads(|| {
            let file = std::fs::File::open(&path)
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Unable to read file: {e}")))?;
            let reader = BufReader::new(file);
            let mut loaded: Self = serde_json::from_reader(reader)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid BM25 JSON: {e}")))?;
            loaded.recompute_stats();
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
            loaded.recompute_stats();
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
            loaded.recompute_stats();
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

    // --- Document management ---

    fn add_document(&mut self, id: String, tokens: Vec<String>, text: String) -> PyResult<()> {
        if id.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("Document id must not be empty."));
        }
        if tokens.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("Tokens must not be empty."));
        }

        if let Some(&old_len) = self.doc_len_map.get(&id) {
            self.total_length -= old_len;
            self.remove_document_from_index(&id);
        }

        let new_len = tokens.len();
        for token in tokens.iter() {
            let target = self.index_map.entry(token.to_string()).or_default();
            *target.entry(id.clone()).or_insert(0) += 1;
        }
        self.doc_len_map.insert(id.clone(), new_len);
        self.doc_texts.insert(id, text);
        self.total_length += new_len;
        self.refresh_average_length();
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
            if let Some(&old_len) = self.doc_len_map.get(&id) {
                self.total_length -= old_len;
                self.remove_document_from_index(&id);
            }

            let new_len = tokens.len();
            for token in tokens.iter() {
                let target = self.index_map.entry(token.to_string()).or_default();
                *target.entry(id.clone()).or_insert(0) += 1;
            }
            self.doc_len_map.insert(id.clone(), new_len);
            self.doc_texts.insert(id, text);
            self.total_length += new_len;
        }

        self.refresh_average_length();
        self.mark_unfrozen();
        Ok(())
    }

    fn remove_document(&mut self, id: String) -> PyResult<()> {
        let &doc_len = self.doc_len_map.get(&id).ok_or_else(|| {
            PyErr::new::<PyKeyError, _>(format!("Document '{}' does not exist.", id))
        })?;
        self.remove_document_from_index(&id);
        self.doc_len_map.remove(&id);
        self.doc_texts.remove(&id);
        self.total_length -= doc_len;
        self.refresh_average_length();
        self.mark_unfrozen();
        Ok(())
    }

    fn remove_documents(&mut self, ids: Vec<String>) -> PyResult<()> {
        for id in &ids {
            if !self.doc_len_map.contains_key(id) {
                return Err(PyErr::new::<PyKeyError, _>(
                    format!("Document '{}' does not exist.", id),
                ));
            }
        }

        for id in &ids {
            if let Some(&doc_len) = self.doc_len_map.get(id) {
                self.remove_document_from_index(id);
                self.doc_len_map.remove(id);
                self.doc_texts.remove(id);
                self.total_length -= doc_len;
            }
        }

        self.refresh_average_length();
        self.mark_unfrozen();
        Ok(())
    }

    fn clear(&mut self) {
        self.index_map.clear();
        self.doc_len_map.clear();
        self.doc_texts.clear();
        self.freeze_map.clear();
        self.total_length = 0;
        self.average_length = 0.0;
        self.is_frozen = false;
    }

    // --- Search ---

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
        self.ensure_frozen().map_err(PyErr::new::<PyRuntimeError, _>)?;
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
        self.ensure_frozen().map_err(PyErr::new::<PyRuntimeError, _>)?;
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

    // --- Parameters ---

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

    // --- Getters / Setters ---

    fn get_document(&self, id: String) -> PyResult<(String, usize)> {
        let text = self
            .doc_texts
            .get(&id)
            .ok_or_else(|| PyErr::new::<PyKeyError, _>(format!("Document '{}' does not exist.", id)))?;
        let doc_len = *self.doc_len_map.get(&id).unwrap_or(&0);
        Ok((text.clone(), doc_len))
    }

    fn get_freeze_map(&self) -> PyResult<HashMap<String, HashMap<String, f32>>> {
        Ok(self.freeze_map.clone())
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
        self.recompute_stats();
        self.mark_unfrozen();
    }

    // --- Utility (legacy compat) ---

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
