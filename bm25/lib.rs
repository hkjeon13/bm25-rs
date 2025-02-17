extern crate pyo3;

use std;
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
struct BM25 {
    index_map: HashMap<String, HashMap<String, u32>>,
    doc_len_map: HashMap<String, usize>,
    doc_texts: HashMap<String, String>,
    freeze_map: HashMap<String, HashMap<String, f32>>,
    k1: f32,
    b: f32,
    average_length: f32,
}


impl BM25 {
    fn calculate_score(&self, tf: f32, df: f32, doc_len: usize) -> f32 {
        let num_docs = self.doc_len_map.len() as f32;
        let k1 = self.k1;
        let b = self.b;
        let average_length = self.average_length;
        (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len as f32 / average_length))) * (((num_docs + 1.0) / (df + 1.0)).ln() + 1.0)
    }

    fn update_freeze_map(&mut self) {
        self.average_length = self.doc_len_map.values().sum::<usize>() as f32 / self.doc_len_map.len() as f32;
        self.freeze_map = self.index_map.iter().map(|(token, doc_freq)| {
            (token.clone(), doc_freq.iter().map(|(doc_id, &tf)| {
                let doc_len = *self.doc_len_map.get(doc_id).unwrap_or(&0);
                let df = doc_freq.len() as f32;
                let score = self.calculate_score(tf as f32, df, doc_len);
                (doc_id.clone(), score)
            }).collect())
        }).collect();
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
        }
    }

    #[classmethod]
    fn load(_cls: &PyType, path: String) -> Self {
        let json_file = std::fs::read_to_string(path).expect("Unable to read file");
        serde_json::from_str(&json_file).unwrap()
    }

    fn save(&self, path: String) {
        let json_file = serde_json::to_string(&self).unwrap();
        std::fs::write(path, json_file).expect("Unable to write file");
    }

    fn get_freeze_map(&self) -> PyResult<HashMap<String, HashMap<String, f32>>> {
        Ok(self.freeze_map.clone())
    }

    fn set_k1(&mut self, k1: f32) {
        if k1 < 0.0 || k1 > 2.0 {
            panic!("k1 must be between 0.0 and 2.0");
        }
        self.k1 = k1;
    }

    fn set_b(&mut self, b: f32) {
        self.b = b;
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
    }

    fn set_doc_len_map(&mut self, doc_len_map: HashMap<String, usize>) {
        self.doc_len_map = doc_len_map;
    }


    fn add_document(&mut self, id: String, tokens: Vec<String>, text: String) {
        for token in tokens.iter() {
            if !self.index_map.contains_key(token) {
                self.index_map.insert(
                    token.to_string(),
                    HashMap::new(),
                );
            }
            let target = self.index_map.get_mut(token).unwrap();
            if !target.contains_key(id.as_str()) {
                target.insert(id.to_string(), 0);
            };

            *target.get_mut(id.as_str()).unwrap() += 1;
        }
        self.doc_len_map.insert(id.to_string(), tokens.len());
        self.doc_texts.insert(id.to_string(), text);
    }

    fn freeze(&mut self) {
        self.update_freeze_map();
    }

    fn search(&self, query_tokens: Vec<String>, n: usize) -> PyResult<Vec<(f32, String, String)>> {
        if self.freeze_map.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Index is not frozen."));
        }

        let scores = query_tokens.iter()
            .filter_map(|token| self.freeze_map.get(token))
            .flat_map(|doc_scores| doc_scores)
            .fold(HashMap::new(), |mut acc, (doc_id, &score)| {
                *acc.entry(doc_id.clone()).or_insert(0.0) += score;
                acc
            });

        let mut results: Vec<_> = scores.into_iter()
            .filter_map(
                |(id, score)|
                    Some((score, id.to_owned(), self.doc_texts.get(&id)?.to_owned()))
            ).collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        Ok(results)
    }


    fn batch_search(&self, tokenized_queries: Vec<Vec<String>>, n: usize) -> PyResult<Vec<Vec<(f32, String, String)>>> {
        Ok(tokenized_queries.par_iter().map(
            |tokenized_query| self.search(tokenized_query.to_vec(), n).unwrap()
        ).collect())
    }

    fn remove_document(&mut self, id: String) {
        // Collect tokens to be modified from index_map
        let tokens_to_modify: Vec<String> = self.index_map.iter()
            .filter(|(_, target)| target.contains_key(&id))
            .map(|(token, _)| token.clone())
            .collect();

        // Perform modifications
        for token in tokens_to_modify {
            self.index_map.get_mut(&token).unwrap().remove(&id);
        }

        self.doc_len_map.remove(&id);
        self.doc_texts.remove(&id);

        // Perform similar steps for freeze_map if it's not empty
        if !self.freeze_map.is_empty() {
            let freeze_tokens_to_modify: Vec<String> = self.freeze_map.iter()
                .filter(|(_, target)| target.contains_key(&id))
                .map(|(token, _)| token.clone())
                .collect();

            for token in freeze_tokens_to_modify {
                self.freeze_map.get_mut(&token).unwrap().remove(&id);
            }
        }
    }
}

#[pymodule]
fn bm25(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}