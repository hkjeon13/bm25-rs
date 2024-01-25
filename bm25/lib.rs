extern crate pyo3;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use serde_derive::{Serialize,Deserialize};
use std::path::Path;
use tqdm_rs;
use counter::Counter;
use rayon::prelude::*;
use pyo3::types::PyType;
use std;

fn _calculate(tf: f32, num_docs: f32, doc_len: usize, average_length: f32, k1: f32, b: f32, df: f32) -> f32 {
    (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len as f32 / average_length))) * (((num_docs as f32 + 1.0) / (df + 1.0)).ln() + 1.0)
}


#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
struct BM25 {
    index_map: HashMap<String, HashMap<String, u32>>,
    doc_len_map: HashMap<String, usize>,
    freeze_map: HashMap<String, HashMap<String, f32>>,
    k1: f32,
    b: f32,
    average_length: f32,
}



#[pymethods]
impl BM25 {
    #[new]
    fn new() -> Self {
        BM25 { index_map: HashMap::new(), doc_len_map: HashMap::new(), freeze_map: HashMap::new(), k1: 1.5, b: 0.75, average_length: 0.0}
    }

    #[classmethod]
    fn load(cls: &PyType, path:String) -> Self {
        let json_file = std::fs::read_to_string(path).expect("Unable to read file");
        serde_json::from_str(&json_file).unwrap()
    }

    fn save(&self, path: String) {
        let json_file = serde_json::to_string(&self).unwrap();
        std::fs::write(path, json_file).expect("Unable to write file");
    }


    fn add_document(&mut self, id: String, document: Vec<String>) {
        for token in document.iter() {
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
        self.doc_len_map.insert(id.to_string(), document.len());
    }

    fn freeze(&mut self) {
        //todo!
        self.average_length = self.doc_len_map.values().sum::<usize>() as f32 / self.doc_len_map.len() as f32;
        self.freeze_map = self.index_map.iter()
            .map(|(k, doc_freq)| (k.to_string(), doc_freq.iter()
                .map(|(dk, dv)|
                    (
                        dk.to_string(),
                        _calculate(
                            *dv as f32,
                            self.doc_len_map.len() as f32,
                            self.doc_len_map.get(dk).unwrap().clone(),
                            self.average_length,
                            self.k1,
                            self.b,
                            doc_freq.len() as f32,
                        )
                    )
                ).collect::<HashMap<String, f32>>())
            ).collect::<HashMap<String, HashMap<String, f32>>>();
    }



    fn search(&self, query_tokens: Vec<String>, n: usize) -> PyResult<Vec<(String, f32)>> {
        if self.freeze_map.len() == 0 {
            panic!("Please freeze the index before searching!");
        }
        let mut scores = HashMap::new();
        for (query, _) in query_tokens.iter().collect::<Counter<_>>().iter(){
            if self.freeze_map.contains_key(query.as_str()){
                let targets = self.freeze_map.get(query.as_str()).unwrap();
                for (doc_id, score) in targets {
                    if !scores.contains_key(doc_id.as_str()) {
                        scores.insert(doc_id.to_string(), 0.0);
                    }
                    *scores.get_mut(doc_id.as_str()).unwrap() += score;
                }
            }
        }
        let mut scores = scores.iter().map(|(k, v)| (k.to_string(), v.to_owned())).collect::<Vec<(String, f32)>>();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);
        Ok(scores)
    }
    

    fn batch_search(&self, tokenized_queries: Vec<Vec<String>>, n: usize) -> PyResult<Vec<Vec<(String, f32)>>> {
        Ok(tokenized_queries.par_iter().map(
            |tokenized_query| self.search(tokenized_query.to_vec(), n).unwrap()
        ).collect())
    }

    fn delete_document(&mut self, id: String) {
        for (token, _) in self.index_map.iter_mut() {
            let target = self.index_map.get_mut(token).unwrap();
            if target.contains_key(id.as_str()) {
                target.remove(id.as_str());
            }
        }
        self.doc_len_map.remove(id.as_str());
        //remove document from freeze_map
        for (token, _) in self.freeze_map.iter_mut() {
            let target = self.freeze_map.get_mut(token).unwrap();
            if target.contains_key(id.as_str()) {
                target.remove(id.as_str());
            }
        }
    }

}

#[pymodule]
fn bm25(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}
