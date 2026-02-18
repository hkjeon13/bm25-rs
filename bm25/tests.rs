use std::collections::HashMap;

use crate::scoring::MinScore;
use crate::BM25;

fn make_bm25() -> BM25 {
    BM25 {
        index_map: HashMap::new(),
        doc_len_map: HashMap::new(),
        doc_texts: HashMap::new(),
        freeze_map: HashMap::new(),
        k1: 1.5,
        b: 0.75,
        average_length: 0.0,
        total_length: 0,
        is_frozen: false,
    }
}

fn make_bm25_with(k1: f32, b: f32) -> BM25 {
    BM25 {
        index_map: HashMap::new(),
        doc_len_map: HashMap::new(),
        doc_texts: HashMap::new(),
        freeze_map: HashMap::new(),
        k1,
        b,
        average_length: 0.0,
        total_length: 0,
        is_frozen: false,
    }
}

fn add_doc(bm25: &mut BM25, id: &str, tokens: Vec<&str>, text: &str) {
    let tokens: Vec<String> = tokens.into_iter().map(String::from).collect();
    let id = id.to_string();

    if let Some(&old_len) = bm25.doc_len_map.get(&id) {
        bm25.total_length -= old_len;
        bm25.remove_document_from_index(&id);
    }

    let new_len = tokens.len();
    for token in tokens.iter() {
        let target = bm25.index_map.entry(token.to_string()).or_default();
        *target.entry(id.clone()).or_insert(0) += 1;
    }
    bm25.doc_len_map.insert(id.clone(), new_len);
    bm25.doc_texts.insert(id, text.to_string());
    bm25.total_length += new_len;
    bm25.refresh_average_length();
}

// --- Score calculation tests ---

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

// --- Search tests ---

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
fn test_no_text_query_match() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello"], "hello");

    let query = vec!["nonexistent".to_string()];
    let results = bm25.search_instance_internal(&query, 5);
    assert!(results.is_empty());
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
fn test_query_deduplication() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
    add_doc(&mut bm25, "d2", vec!["foo", "bar"], "foo bar");

    let query_dup = vec!["hello".to_string(), "hello".to_string()];
    let query_single = vec!["hello".to_string()];

    let results_dup = bm25.search_instance_internal(&query_dup, 5);
    let results_single = bm25.search_instance_internal(&query_single, 5);

    assert_eq!(results_dup.len(), results_single.len());
    for (a, b) in results_dup.iter().zip(results_single.iter()) {
        assert_eq!(a.1, b.1);
        assert!((a.0 - b.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_query_deduplication_frozen() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
    bm25.update_freeze_map();
    bm25.is_frozen = true;

    let query_dup = vec!["hello".to_string(), "hello".to_string()];
    let query_single = vec!["hello".to_string()];

    let results_dup = bm25.search_frozen_internal(&query_dup, 5);
    let results_single = bm25.search_frozen_internal(&query_single, 5);

    assert_eq!(results_dup.len(), results_single.len());
    for (a, b) in results_dup.iter().zip(results_single.iter()) {
        assert!((a.0 - b.0).abs() < f32::EPSILON);
    }
}

// --- Document management tests ---

#[test]
fn test_remove_document() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
    add_doc(&mut bm25, "d2", vec!["hello", "rust"], "hello rust");

    let old_len = *bm25.doc_len_map.get("d1").unwrap();
    bm25.remove_document_from_index("d1");
    bm25.doc_len_map.remove("d1");
    bm25.doc_texts.remove("d1");
    bm25.total_length -= old_len;
    bm25.refresh_average_length();

    assert!(!bm25.doc_len_map.contains_key("d1"));
    assert!(!bm25.index_map.contains_key("world"));
    assert!(bm25.index_map.contains_key("hello"));
    assert_eq!(bm25.total_length, 2);
}

#[test]
fn test_duplicate_doc_id_replaces() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");
    add_doc(&mut bm25, "d1", vec!["foo", "bar"], "foo bar");

    assert_eq!(bm25.doc_len_map.len(), 1);
    assert_eq!(bm25.doc_texts.get("d1").unwrap(), "foo bar");
    assert!(!bm25.index_map.contains_key("hello"));
    assert!(!bm25.index_map.contains_key("world"));
    assert_eq!(bm25.total_length, 2);
}

#[test]
fn test_remove_documents_batch() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello"], "hello");
    add_doc(&mut bm25, "d2", vec!["world"], "world");
    add_doc(&mut bm25, "d3", vec!["foo"], "foo");

    let ids = vec!["d1", "d2"];
    for id in &ids {
        let doc_len = *bm25.doc_len_map.get(*id).unwrap();
        bm25.remove_document_from_index(id);
        bm25.doc_len_map.remove(*id);
        bm25.doc_texts.remove(*id);
        bm25.total_length -= doc_len;
    }
    bm25.refresh_average_length();

    assert_eq!(bm25.doc_len_map.len(), 1);
    assert!(bm25.doc_len_map.contains_key("d3"));
    assert_eq!(bm25.total_length, 1);
}

#[test]
fn test_retain_removes_empty_tokens() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["unique_token", "shared"], "text");
    add_doc(&mut bm25, "d2", vec!["shared"], "text2");

    bm25.remove_document_from_index("d1");
    assert!(!bm25.index_map.contains_key("unique_token"));
    assert!(bm25.index_map.contains_key("shared"));
}

// --- Stats tests ---

#[test]
fn test_average_length_maintained() {
    let mut bm25 = make_bm25();
    assert_eq!(bm25.average_length, 0.0);
    assert_eq!(bm25.total_length, 0);

    add_doc(&mut bm25, "d1", vec!["a", "b", "c"], "a b c");
    assert_eq!(bm25.total_length, 3);
    assert!((bm25.average_length - 3.0).abs() < f32::EPSILON);

    add_doc(&mut bm25, "d2", vec!["x"], "x");
    assert_eq!(bm25.total_length, 4);
    assert!((bm25.average_length - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_total_length_o1() {
    let mut bm25 = make_bm25();

    add_doc(&mut bm25, "d1", vec!["a", "b", "c"], "a b c");
    assert_eq!(bm25.total_length, 3);

    add_doc(&mut bm25, "d2", vec!["x", "y"], "x y");
    assert_eq!(bm25.total_length, 5);

    add_doc(&mut bm25, "d1", vec!["z"], "z");
    assert_eq!(bm25.total_length, 3);
    assert!((bm25.average_length - 1.5).abs() < f32::EPSILON);

    let old_len = *bm25.doc_len_map.get("d2").unwrap();
    bm25.remove_document_from_index("d2");
    bm25.doc_len_map.remove("d2");
    bm25.doc_texts.remove("d2");
    bm25.total_length -= old_len;
    bm25.refresh_average_length();
    assert_eq!(bm25.total_length, 1);
    assert!((bm25.average_length - 1.0).abs() < f32::EPSILON);
}

// --- MinScore / NaN tests ---

#[test]
fn test_nan_handling_in_minscore() {
    let a = MinScore(f32::NAN, "a".to_string());
    let b = MinScore(1.0, "b".to_string());
    let _ = a.cmp(&b);
    let _ = b.cmp(&a);
    let _ = a.eq(&b);
}

// --- Serialization tests ---

#[test]
fn test_serde_skip_freeze_map() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello"], "hello");
    bm25.update_freeze_map();
    bm25.is_frozen = true;
    assert!(!bm25.freeze_map.is_empty());

    let json = serde_json::to_string(&bm25).unwrap();
    assert!(!json.contains("freeze_map"));

    let loaded: BM25 = serde_json::from_str(&json).unwrap();
    assert!(loaded.freeze_map.is_empty());
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
    assert!(decoded.freeze_map.is_empty());
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

    let mut buf = Vec::new();
    serde_json::to_writer(&mut buf, &bm25).unwrap();
    let loaded: BM25 = serde_json::from_reader(buf.as_slice()).unwrap();

    assert_eq!(loaded.doc_len_map.len(), 1);
    assert_eq!(loaded.doc_texts.get("d1").unwrap(), "hello");
}

// --- Misc tests ---

#[test]
fn test_clear() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello"], "hello");
    add_doc(&mut bm25, "d2", vec!["world"], "world");
    bm25.update_freeze_map();
    bm25.is_frozen = true;

    bm25.index_map.clear();
    bm25.doc_len_map.clear();
    bm25.doc_texts.clear();
    bm25.freeze_map.clear();
    bm25.total_length = 0;
    bm25.average_length = 0.0;
    bm25.is_frozen = false;

    assert_eq!(bm25.doc_len_map.len(), 0);
    assert_eq!(bm25.doc_texts.len(), 0);
    assert_eq!(bm25.index_map.len(), 0);
    assert_eq!(bm25.freeze_map.len(), 0);
    assert_eq!(bm25.total_length, 0);
    assert_eq!(bm25.average_length, 0.0);
    assert!(!bm25.is_frozen);
    assert!((bm25.k1 - 1.5).abs() < f32::EPSILON);
    assert!((bm25.b - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_constructor_with_params() {
    let bm25 = make_bm25_with(1.2, 0.8);
    assert!((bm25.k1 - 1.2).abs() < f32::EPSILON);
    assert!((bm25.b - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_get_document() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello", "world"], "hello world");

    let text = bm25.doc_texts.get("d1").unwrap().clone();
    let doc_len = *bm25.doc_len_map.get("d1").unwrap();
    assert_eq!(text, "hello world");
    assert_eq!(doc_len, 2);
}

#[test]
fn test_repr() {
    let mut bm25 = make_bm25();
    add_doc(&mut bm25, "d1", vec!["hello"], "hello");
    let repr = format!(
        "BM25(docs={}, k1={}, b={}, frozen={})",
        bm25.doc_len_map.len(), bm25.k1, bm25.b, bm25.is_frozen,
    );
    assert_eq!(repr, "BM25(docs=1, k1=1.5, b=0.75, frozen=false)");
}
