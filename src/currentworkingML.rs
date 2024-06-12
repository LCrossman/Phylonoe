use pyo3::{Py, Python};
use numpy::{PyReadonlyArray1, PyArrayDyn};
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SVD;
use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::collections::HashMap;
use pyo3::types::IntoPyDict;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyTuple;
use std::fs::File;
use std::io::{BufRead, BufReader};
use pyo3::types::PyDict;
use pyo3::types::PyList;
use std::error::Error;
use pyo3::py_run;
use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use bio::io::fasta::Record;
use bio::io::fasta::{FastaRead, Reader};
use clap::Parser;


#[derive(Parser,Default,Debug)]
#[clap(author="LCrossman",version,about="Machine learning from sequences and labels, inputs are file of filenames separated by commas and labels are the names of the classes separated by commas.")]
struct Arguments {
   #[clap(short, long,default_value="1")]
   threads: String,
   #[clap(short,long)]
   inputs: String,
   #[clap(short,long)]
   labels: String,
}


struct ChunksReader {
    chunks: Vec<Vec<u8>>,
    current_chunk: usize,
}

impl Read for ChunksReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
       if self.current_chunk < self.chunks.len() {
          let remaining = &self.chunks[self.current_chunk];
          let to_copy = std::cmp::min(buf.len(), remaining.len());
          buf[..to_copy].copy_from_slice(&remaining[..to_copy]);
          self.current_chunk +=1;
          Ok(to_copy)
       } else {
          Ok(0)
       }
    }
}

pub fn shred() -> Result<()> {
   let args = Arguments::parse();
   let files: Vec<String> = args.filename.split(',').map(|f| f.to_string()).collect();
   for file in files {
      let outfname = format!("{}_20.proc", file.to_string());
      let path = File::open(&file).expect("error opening file");
      let mut reader = Reader::new(path);
      let mut writes = File::create(&outfname).expect("opening creating write file");
      let mut writer = BufWriter::new(writes);
      let word_vec: Vec<String> = Vec::new();
      for result in reader.records() {
           let record = result.expect("error during fasta record parsing");
           let seq = String::from_utf8_lossy(record.seq());
           for word in seq.chars().collect::<Vec<char>>().chunks(20) {
               let mut word: String = word.into_iter().collect();
               if let Some('*') = word.chars().last() {
                  word.pop();
                  }
               if word.len() < 20 { 
                  println!("lesser word {:?}", word); 
                  } else { 
                  word_vec.push(word);
                  }
               }
           }
           for wor in &word_vec {
              write!(writer, "{} ", wor).unwrap();
              writer.flush();
              }
   }
   Ok(())
}

   
//#[pyfunction]
//fn plot_feat_importances(
//    importances: &PyArray1<f64>,
//    indices: &PyArray1<usize>,
//    std: &PyArray1<f64>,
//    x: &PyArray2<f64>,
     //) -> PyResult<()> {
   // let plt = Python::acquire_gil().python().import("matplotlib.pyplot")?;

//    plt.call_method0("figure", ())?;
//    plt.call_method1("title", ("Feature importances",))?;
//    plt.call_method1(
//        "bar",
//        (
//            (0..x.shape()[1]).collect::<Vec<_>>(),
//            importances.get_indices(indices),
//        ),
//    )?;
//    plt.call_method1(
//        "errorbar",
//        (
//            (0..x.shape()[1]).collect::<Vec<_>>(),
//            importances.get_indices(indices),
  //          std.get_indices(indices),
//        ),
//    )?;
//    plt.call_method1("xticks", ((0..x.shape()[1]).collect::<Vec<_>>(), indices))?;
//    plt.call_method0("xlim", ((-1, x.shape()[1]),))?;
//    plt.call_method0("show", ())?;

//    Ok(())
          //}


//#[pyfunction]
//pub fn select_k_importance(model: &PyAny, x: &PyArray2<f64>, k: usize) -> PyResult<Py<PyArray2<f64>>> {
    // Get feature importances from the model
//    let feature_importances: &PyArray1<f64> = model.getattr("feature_importances_")?.downcast()?;
//    let mut df = DataFrame::new(vec![]).unwrap();
//    for col_idx in 0..x.ncols() {
//          let series = Series::new("", x.column(col_idx).to_vec());
//          df.add_column(series).unwrap();
//          }
//    let importance_series = Series::new("importance", feature_importances);
//    let df_with_importances = df.with_column(importance_series)?;
//    let sorted_df = df_with_importances.sort("importance", false)?;
//    let selected_df = sorted_df.select_at_idx((0..k).collect())?;
//    let selected_x = selected_df.to_ndarray::<Float64Type>()?;
//    Ok(selected_x.to_owned())
     //}


//#[pyfunction]
//fn plot_confusion_matrix(
//    cm: &PyArray2<f64>,
//    classes: Vec<String>,
//    normalize: bool,
//    title: &str,
//    cmap: &str,
     //) -> PyResult<()> {
    //let plt = Python::acquire_gil().python().import("matplotlib.pyplot")?;
//    let np = Python::acquire_gil().python().import("numpy")?;
//    let itertools = Python::acquire_gil().python().import("itertools")?;

//    plt.call_method1("imshow", (cm,))?;
//    plt.call_method1("title", (title,))?;
//    plt.call_method0("colorbar", ())?;

  //  let tick_marks = np.call_method1("arange", (classes.len(),))?;
 //   plt.call_method2("xticks", (tick_marks.clone(), &classes,))?;
//    plt.call_method2("yticks", (tick_marks.clone(), &classes,))?;

//    if normalize {
 //       let cm_normalized = cm / cm.sum_axis(Axis(1), true);
 //       println!("Normalized confusion matrix");
 //       println!("{}", cm_normalized);
 //   } else {
  //      println!("Confusion matrix, without normalization");
 //       println!("{}", cm);
 //   }

   // let thresh = cm.max() / 2.0;
 //   for (i, j) in itertools.call_method0("product", (cm.shape()[0], cm.shape()[1]))? {
 //       let text_color = if cm.get((i, j)) > thresh { "white" } else { "black" };
 //       plt.call_method1(
 //           "text",
 //           (
//                j,
 //               i,
 //               cm.get((i, j)),
 //               [("horizontalalignment", "center"), ("color", text_color)],
  //          ),
  //      )?;
  //  }

//    plt.call_method0("tight_layout", ())?;
//    plt.call_method1("ylabel", ("True label",))?;
//    plt.call_method1("xlabel", ("Predicted label",))?;

//    Ok(())
          //}

//#[pyclass]
//struct Parameters {
//     loss: &str,
 //    penalty: &str,
//     n_iter: u64,
//     alpha: f64,
//     fit_intercept: bool,
//     }

//impl<'de> Deserialize<'de> for Parameters<'_> {
//     fn deserialize<D>(deserializer: D) -> Result<Parameters, <D as serde::Deserializer<'de>>::Error>> {
//    where
//        D: serde::Deserializer<'de>,
//    {
//        let map: HashMap<String, serde_json::Value> = Deserialize::deserialize(deserializer)?;
//        let loss = map.get("loss").ok_or_else(|| serde::de::Error::custom("missing loss"))?.as_str().ok_or_else(|| serde::de::Error::custom("loss is not a string"))?.to_owned();
//        let penalty = map.get("penalty").ok_or_else(|| serde::de::Error::custom("missing penalty"))?.as_str().ok_or_else(|| serde::de::Error::custom("penalty is not a string"))?.to_owned();
//        let n_iter = map.get("n_iter").ok_or_else(|| serde::de::Error::custom("missing loss"))?.as_u64().ok_or_else(|| serde::de::Error::custom("loss is not a string"))? as u64;
//        let alpha = map.get("alpha").ok_or_else(|| serde::de::Error::custom("missing loss"))?.as_f64().ok_or_else(|| serde::de::Error::custom("loss is not a string"))? as f64;
//        let fit_intercept = map.get("fit_intercept").ok_or_else(|| serde::de::Error::custom("missing fit_intercept"))?.as_bool().ok_or_else(|| serde::de::Error::custom("fit_intercept is not a bool"))?;
        
//        Ok(Parameters { loss, penalty, n_iter, alpha, fit_intercept })
//    }
//   }/
//}

//#[pymethods]
//impl Parameters<'_> {
//    fn new(loss: String, penalty: String, n_iter: u64, alpha: f64, fit_intercept: bool) -> Self {
//        Parameters { loss: &loss, penalty: &penalty, n_iter: n_iter, alpha: alpha, fit_intercept: fit_intercept }
//    }
     //}

//#[pyfunction]
//fn benchmark(_clf_class: PyObject, _params: PyObject, _name: &str) -> PyResult<()> {
 //   // Python imports
   // let sys = Python::import(py, "sys")?;
 //   let time = Python::import(py, "time")?;
//    let plt = Python::import(py, "matplotlib.pyplot")?;
  //  let np = Python::import(py, "numpy")?;
  //  let metrics = Python::import(py, "sklearn.metrics")?;
  //  let params = [ "hinge", "l2", 50, 0.00001, true ];
    // Print parameters
  //  println!("parameters: {:?}", params);

    // Measure time
  //  let t0 = time.call_method0("time")?;

    // Create and fit classifier
  //  let clf = Python::import(py, "sklearn.naive_bayes")?.call_method1(
  //      "MultinomialNB", &[("fit_prior", true), ("class_prior", None)])?;
  //  clf.call_method1("fit", (features_train, labels_train))?;

    // Print time taken
  //  println!("done in {}s", time.call_method0("time")? - t0);

    // Check if the classifier has coefficients
  //  if clf.hasattr("coef_")? {
  //      let coef = clf.getattr("coef_")?;
 //       let nonzero_coef_percentage = (np.call_method1("__ne__", (coef, 0))? as f64).mean() * 100.0;
   //     println!("Percentage of non zeros coef: {}", nonzero_coef_percentage);
   // }

    // Predict outcomes
   // println!("Predicting the outcomes of the testing set");
  //  let t0 = time.call_method0("time")?;
  //  let pred = clf.call_method0("predict", (features_test,))?;
  //  println!("done in {}s", time.call_method0("time")? - t0);

    // Calculate confusion matrix
  //  let confusion_matrix = metrics.call_method1("confusion_matrix", (features_test, pred))?;
 //   println!("Confusion matrix:");
 //   println!("{}", confusion_matrix);

    // Calculate precision
 //   let diagonal = confusion_matrix.getattr("diagonal")()?;
 //   let precision = diagonal.call_method0("sum")? / confusion_matrix.call_method0("sum", (Axis(1),))?;
 //   println!("{}", precision);

    // Show confusion matrix
 //   plt.call_method0("matshow", (confusion_matrix,))?;
 //   plt.setattr("title", "Confusion matrix of the %s classifier".to_owned() + _name)?;
 //   plt.call_method0("colorbar", ())?;

//    Ok(())
//}
    
//#[pyfunction]  
//fn doPCA(data: &PyArray2<f64>) -> PyResult<Vec<f64>> {
//    let data_array = data.as_array();
//    let svd = data_array.svd(true, true)?;
//    let first_two_components = svd.vt.slice(s![..2, ..]).to_vec();
//    Ok(first_two_components)/
     //}


//#[pyfunction]
//pub fn to_categorical(py: Python, y: i32) -> PyResult<PyArray1<> {
    //Converts a class vector (integers) to binary class matrix.
    //E.g. for use with categorical_crossentropy.
    //# Arguments
    //    y: class vector to be converted into a matrix
    //        (integers from 0 to num_classes).
    //    num_classes: total number of classes.
    //# Returns
    //    A binary matrix representation of the input.
//    let categorical: PyArray1 = PyArrayDyn::arange(py, y);
//    py_run!(py, categorical, r#"
//       import numpy as np
//       num_classes = None
//       y = np.array(y, dtype='int').ravel()
 //      if not num_classes:
//           num_classes = np.max(y) + 1
//       n = y.shape[0]
//       categorical = np.zeros((n, num_classes))
 //      categorical[np.arange(n), y] = 1
              //           "#");
  //  Ok(categorical)
//}


pub fn parse_out_text(mut f: &std::path::Path) -> Vec<String> {
    let file = File::open(f).expect("failed to open file");
    let mut reader = BufReader::new(file);
    let mut words: Vec<String> = Vec::new();
    for line in reader.lines() {
          let text_string: Vec<_> = line.expect("no line").trim_end().split_whitespace().map(|s| s.to_string()).collect();
          for element in text_string.iter() {
              words.push(element.to_string());
              }
          }
    words
}


//#[pyfunction]
//pub fn show_most_informative_features(vectorizer: &PyAny, clf: &PyAny, n: usize) -> PyResult<()> {
    // Get feature names from vectorizer
//    let feature_names: Vec<String> = vectorizer.call_method0("get_feature_names")?.extract()?;
    // Get coefficients from clf
//    let coef: Vec<f64> = clf.getattr("coef_")?.get_item(0)?.extract()?;
    // Zip coefficients with feature names and sort by coefficients
//    let mut coefs_with_fns: Vec<(&f64, String)> = coef.iter().zip(feature_names.iter()).collect();
//    coefs_with_fns.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Get top n features with positive and negative coefficients
//    let top = coefs_with_fns.iter().take(n).zip(coefs_with_fns.iter().rev().take(n));
    // Print the results
//    for ((&coef_1, &fn_1), (&coef_2, fn_2)) in top {
//        println!(
//            "\t{:.4}\t{:15}\t\t{:.4}\t{:15}",
//            coef_1, fn_1, coef_2, fn_2
//        );
//    }
//    Ok(())
//}

#[pyfunction]
fn metrics(py: Python, result: &PyTuple, labels: Vec<String>) -> PyResult<()> {
    let res = String::new();
    let predicted = result.get_item(0).unwrap();
    let labels_test = result.get_item(1).unwrap();
    py_run!(py, res predicted labels_test labels, r#"
def plot_confusion_matrix(cm, classes,title='Confusion matrix',normalize=False):
    import matplotlib.pyplot as plt
    from itertools import product
    import itertools
    import numpy as np
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
print("labels test", labels_test)
accuracy = accuracy_score(predicted, labels_test)
print("accuracy = ", accuracy)
num_classes=len(labels)
#y_categorical = to_categorical(mlb_test, num_classes)
#pred_categorical = to_categorical(predicted, num_classes)
#cnf_matrix = confusion_matrix(pred_categorical.argmax(1), y_categorical.argmax(1))
cnf_matrix = confusion_matrix(predicted, labels_test)
np.set_printoptions(precision=2)
#//class_names = ['1','2','3','4'] # alter to integer format to fit the regressor
class_names = labels
print(cnf_matrix)
cmap = plt.cm.Blues
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()"#);
     Ok(())
}

pub fn decrypt_large_file<'a>(
    source_file_path: &'a PathBuf, k: &'a [u8; 32], n: &'a [u8; 7],) -> Result<Vec<Vec<u8>>, anyhow::Error> {
    let aead = ChaCha20Poly1305::new(k.as_ref().into());
    let mut stream_decryptor = DecryptorBE32::from_aead(aead, n.as_ref().into());
    const BUFFER_LEN: usize = 500+16;
    let mut buffer = [0u8; BUFFER_LEN];
    let mut chunks_reader: Vec<Vec<u8>> = Vec::new();
    let mut source_file = std::fs::File::open(source_file_path).expect("source file not open");
    loop {
       let read_count = source_file.read(&mut buffer).unwrap();
       if read_count == BUFFER_LEN {
          let ciphertext = stream_decryptor
             .decrypt_next(buffer.as_slice())
             .map_err(|err| anyhow!("decrypting file error {}", err))?;
          chunks_reader.push(ciphertext);
       } else {
          let ciphertext = stream_decryptor
             .decrypt_last(&buffer[..read_count])
             .map_err(|err| anyhow!("decrypting file error {}", err))?;
          chunks_reader.push(ciphertext);
          break;
      }
    }
    Ok(chunks_reader)
}
            
#[pyfunction]
pub fn preprocess(inputs: Vec<String>, labels: Vec<String>, fileid: HashMap<String,String>, nonHash: HashMap<String,String>, k2: &[u8; 32]) -> PyResult<(Vec<String>, Vec<Vec<String>>)> {
    // Open files of filenames .fofn
    let zipped: Vec<_> = labels.iter().zip(inputs.iter()).collect();
    let mut from_auth: Vec<String> = Vec::new();
    let mut word_data: Vec<Vec<String>> = Vec::new();
    // Read data from files and preprocess
    for (name, from_dat) in zipped {
        let newpath = &fileid[from_dat];
	let mut w = File::create(wf).expect("unable to create file");
	let mut seqs: HashMap<String, String> = HashMap::new();
	let n2 = &nonHash[newpath];
	let seqchunks_reader = decrypt_large_file(&newpath, &k2, &n2);
	let seqchunks_flattened: Vec<u8> = seqchunks_reader.unwrap().as_slice().into_iter().flatten().map(|&y| y).collect();
	let seq_cursor = Cursor::new(&seqchunks_flattened[..]);
	let buf_reader = BufReader::new(seq_cursor);
       // let path = std::path::Path::new(&newpath);
      //	let file = File::open(&path).expect("failed to read file lines");
       //	let reader = BufReader::new(file);
        for line in buf_reader.lines() {
            let text = parse_out_text(std::path::Path::new(&line.unwrap()));
            word_data.push(text);
            from_auth.push(name.to_string());
        }
    }
    Ok((from_auth, word_data))
}

#[pyfunction]
pub fn process_data(py: Python, inputs: Vec<String>, labels: Vec<String>) -> PyResult<&PyTuple> {
    let feature_names: Vec<String> = vec![];
    let vectorizer: Vec<String> = vec![];
    let features_train_transformed: Vec<String> = vec![];
    let features_test_transformed: Vec<String> = vec![];
    let labels_train: Vec<String> = vec![];
    let pred = PyList::empty(py);
    let lab = PyList::empty(py);
    let result = PyTuple::new(py, &[pred, lab]);
    //let result = PyTuple::new(py, &[feature_names, vectorizer, features_train_transformed, features_test_transformed, labels_train, labels_test]);
    println!("this is rust with result as initiatied {:?}", result);
    //let result = PyTuple::new(py, 0);
    // Read data
    let Ok((from_auth, word_data)) = preprocess(inputs, labels.clone()) else { panic!("preprocessing was not carried out"); };
    //let from_auth_py = [("from_auth_py", from_auth)].into_py_dict(py);
    //let word_data_py = [("word_data_py", word_data)].into_py_dict(py);
     //py_run!(py, result from_auth word_data, r#"
    let processing_data = PyModule::from_code(py, r#"
def process_data_py(result, from_auth, word_data, labels):
    
       print("this is process_data python")
       print("train_test_splitting...")
       print("this is from auth in python", from_auth)
       from sklearn.model_selection import train_test_split
       from sklearn import neighbors
       import numpy as np
       from time import time
       from scipy import sparse, linalg, stats
       from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator
       #from scipy.sparse import issparse as sp
       from sklearn.neighbors import NearestNeighbors
       from sklearn.preprocessing import Normalizer, OneHotEncoder
       from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
       from sklearn.decomposition import TruncatedSVD
       from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
       from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor
       from sklearn.metrics import confusion_matrix
       from sklearn.pipeline import make_pipeline
       from sklearn.metrics import classification_report
       from sklearn.preprocessing import Normalizer
       import matplotlib.pyplot as plt
       from sklearn.feature_selection import SelectPercentile, f_classif, chi2
       features_train, features_test, labels_train, labels_test = train_test_split(word_data, from_auth, test_size=0.25, random_state=42)
       print("splitting finished...")
       print("vectorizing...")
       vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True,smooth_idf=True, tokenizer=lambda doc: doc, lowercase=False)
       print("vectorizing finished...")
       print("transforming...")
       print(type(features_test))
       features_train_transformed = vectorizer.fit_transform(features_train)
       print("training transform finished...")
       print("test features doing...")
       features_test_transformed = vectorizer.transform(features_test)
       print("test features done")
       print("done normalizing")
       print("dealing with features_names")
       feature_names = vectorizer.get_feature_names_out()
       selector = SelectPercentile(score_func=chi2, percentile=33)
       print("selecting done, fitting")
       selector.fit(features_train_transformed, labels_train)
       print("dealing with feature_names")
       if len(feature_names) > 1:
           feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
           feature_names = np.asarray(feature_names)
       print(feature_names.shape)
       print("selected features", feature_names)
       print("fitting done")
       print("creating the array")
       features_train_transformed = selector.transform(features_train_transformed)
       #assert sp.sparse(features_train_transformed)
       features_test_transformed = selector.transform(features_test_transformed)
       #assert sp.sparse(features_test_transformed)
       print(features_test_transformed)
       print("labels train", labels_train)
       print("labels test", labels_test)
       result = (feature_names, vectorizer, features_train_transformed, features_test_transformed, labels_train, labels_test)
       print("this is fit_extratrees python")
       print("preprocessing concluded", len(features_train), len(features_test))
       print(feature_names)
       feat = feature_names
       #d = {}
       #d = {'1':1,'2':1,'3':100,'4':50}
       t0 = time()
       from sklearn import tree
       #from sklearn.preprocessing import MultiLabelBinarizer # to fit MultiLabelBinarizer for Regression 
       #mlb = MultiLabelBinarizer()
       #mlb_train = mlb.fit_transform(labels_train)
       #mlb_test = mlb.fit_transform(labels_test)
       #clf = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1) # to fit the biofilm categories as a sliding numerical scale instead
       clf = ExtraTreesClassifier(n_estimators=1500) #gave the best results on accuracy, better than random forests
       #clf = DecisionTreeClassifier(max_depth=15) # to fit a single decision tree
       clf.fit(features_train_transformed, labels_train)
       print("training time:", round(time()-t0,3),"s")
       class_names = labels  # ['1','2','3','4'] # alter to integer to fit the regressor
       t1= time()
       predicted = clf.predict(features_test_transformed)
       print(predicted)
       print("predicting time:", round(time()-t1,3),"s")
       result = (predicted, labels_test)
       import pydotplus
       #exporting 10 alternative trees from the extratrees
       i_tree = 0
       print("saving 10 decision tree examples from extratrees")
       for tree_in_forest in clf.estimators_:
            if i_tree < 6:
                with open('tree_' + str(i_tree) + "dot", 'w') as my_file:
                    dot_data = tree.export_graphviz(tree_in_forest, out_file=None, feature_names = feature_names, class_names=class_names)
                    graph = pydotplus.graph_from_dot_data(dot_data)
                    graph.write_pdf("Pfam__1234"+str(i_tree)+".pdf")
                    i_tree+=1

       importances = clf.feature_importances_
       indices = np.argsort(importances)[::-1]
       print("top 100 feature ranking:")
       for f in range(len(importances[indices])):
              if f < 101: 
                  print("%d. feature %s (%f)" %(f+1, feature_names[indices[f]], importances[indices[f]]))
       return result"#, "process_data_py.py", "process_data_py")?;
   let result: &PyTuple = processing_data.getattr("process_data_py")?.call1((result, from_auth, word_data, labels.clone()))?.extract()?;
   println!("result is now {:?}", &result);
   let metric = metrics(py, result, labels);
   Ok(result)
}

//#[pyfunction]
//pub fn show_top25(py: Python, classifier: PyResult<PyObject>, vectorizer: PyResult<PyObject>, categories: PyResult<Vec<String>>) -> PyResult<()> {
//    let answer = PyResult::new(&PyAny, PyErr);
//    py_run!(py, answer, r#"
//        feature_names = np.asarray(vectorizer.get_feature_names())
//        for i, category in enumerate(categories):
//             top25 = np.argsort(classifier.coef_[i])[-25:]
//             print("%s: %s" % (category, " ".join(feature_names[top25])))
//    "#);
//    Ok(())
//}


#[pyfunction]
pub fn fit_extratrees(py: Python, inputs: Vec<String>, labels: Vec<String>) -> &PyModule {
    let results = process_data(py, inputs, labels);
    println!("results is {:?}", results);
    let answer = PyModule::from_code(py, r#"
print("this is the first part of python")
#feature_names, vectorizer, features_train, features_test, labels_train, labels_test = process_data(py)
"#, "fit_extratrees.py", "fit_extratrees",).unwrap();
    answer
}


pub fn mymodule(py: Python<'_>, inputs: Vec<String>, labels: Vec<String>) -> PyResult<()> {
   //let mymodule = PyModule::new(py, "mymodule");
   //mymodule.expect("some sort of weird rusty python error").add_function(wrap_pyfunction!(fit_extratrees(py), mymodule.as_ref().unwrap()).expect("expecting again"));
   fit_extratrees(py, inputs, labels);
   Ok(())
}

fn main() -> PyResult<()> {
   let args = Arguments::parse();
   println!("args are parsed {:?}", &args);
   let input = args.inputs;
   let label = args.labels;
   let inputs: Vec<String> = input.split(',').map(|s| s.to_string()).collect();
   let labels: Vec<String> = label.split(',').map(|l| l.to_string()).collect();
   Python::with_gil(|py| {
        mymodule(py, inputs.to_vec(), labels.to_vec())
        });
   Ok(())
}


        

//###uncomment from this section to get plots of top feature importances
//#show_top25(clf, vectorizer, class_names)
//#importances = clf.feature_importances_
//#std = np.std(clf.feature_importances_,axis=0)
//#indices = np.argsort(importances)[::-1]
//#std = np.std([importan for importan in importances])
//#importances = selectKImportance(clf,features_train,25)
//#importances = clf.feature_importances_
//#indices = np.argsort(importances)
//#features = features_train.columns
//#f,ax = plt.subplots(figsize=(11,9))
//#plt.title("Feature ranking", fontsize=20)
//#plt.bar(range(importances.shape[0]),importances[indices], color="b", align="center")
//#plt.xticks(range(importances.shape[0], indices))
//#plt.xlim([-1, importances.shape[0]])
//#plt.ylabel("importance", fontsize=18)
//#plt.xlabel("index of the feature", fontsize=18)

//#//#//#//#//#top_ranked_features = sorted(enumerate(clf.feature_importances_),key=lambda x:x[1], reverse=True)
//#//#//#//#//#print top_ranked_features
//#//#//#//#//#outfile = open("top_rank.txt", 'w')
//#//#//#//#//#top_ranked_features_indices = map(list, zip(*top_ranked_features))[0]
//#//#//#//#//#top_ranked_features_p = map(list, zip(*top_ranked_features))[1]
//#//#//#//#//#print len(top_ranked_features_indices), "len"
//#//#//#//#//#for fe in top_ranked_features_indices:
   //#//#//#//#//# print feat[fe], top_ranked_features_p[fe]
    //#//#//#//#//#outfile.write("{}\n".format(feat[fe]))
//#    print("%s. feature %d (%f)"%(feat[fe+1], top_ranked_features_indices[fe], importances[top_ranked_features_indices[fe]]))
//#plt.figure()
//#plt.title("Feature importances")
//#plt.barh(range(len(top_ranked_features_indices)), importances[top_ranked_features_indices], color="r", align="center")
//#plt.yticks(range(len(top_ranked_features_indices)), top_ranked_features[top_ranked_features_indices])
//#plt.xlim([-1, features_train.shape[1]])
//#plt.show()
//#print feat[top_ranked_features_indices]
//#outfile.write("{}\n".format([fe for fe in feat[top_ranked_features_indices]]))
//#for feature_pvalue in zip(feat[top_ranked_features_indices],clf.pvalues_[top_ranked_features_indices]):
//#    print feature_pvalue

//#plt.figure(1)
//#plt.title('Feature importances')
//#plt.barh(range(len(indices)), feat[indices], color='b', align='center')
//#plt.yticks(range(len(indices)), feat[indices])
//#plt.xlabel('Relative importance')
//#plt.show()
//#print "features sorted by their score:"
//#print importances

//#print("Feature ranking:")
//#for f in range(10):
//#    print("%d. feature %d (%f)"% (f + 1, indices[f], importances[indices[f]]))

//#plt.figure()
//#plt.title("Feature importances")
//#plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
//#plt.xticks(range(10), indices)
//#plt.xlim([-1,10])
//#plt.show()


//#[pyfunction]
//fn save_trees(py: Python) -> PyResult<()> {
//    let answer = String::new();
//    py_run!(py, answer, r#"
//        import pydotplus
 //       //#exporting 10 alternative trees from the extratrees
 //       i_tree = 0
 //       for tree_in_forest in clf.estimators_:
  //          if i_tree < 11:
  //              with open('tree_' + str(i_tree) + "dot", 'w') as my_file:
    //                dot_data = tree.export_graphviz(tree_in_forest, out_file=None, feature_names = feature_names, class_names=class_names)
      //              graph = pydotplus.graph_from_dot_data(dot_data)
        //            graph.write_pdf("Pfam__1234"+str(i_tree)+".pdf")
          //          i_tree+=1

//        importances = clf.feature_importances_
 //       indices = np.argsort(importances)[::-1]
  //      print("feature ranking:")
  //      for f in xrange(len(importances[indices])):
   //           print("%d. feature %s (%f)" %(f+1, feature_names[indices[f]], importances[indices[f]]))
 //   "#);
 //    Ok(())
//}


//with open("iris.dot", 'w') as fa:
//    fa = tree.export_graphviz(clf, out_file=fa)

//#plotfeatImportances(importances, indices, std, features_train)

