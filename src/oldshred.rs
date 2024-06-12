use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::io::Write;
use std::error::Error;
use bio::io::fasta::Record;
use bio::io::fasta::{FastaRead, Reader};
use clap::Parser;


//#[derive(Parser,Default,Debug)]
//#[clap(author="LCrossman",version,about="Machine learning from sequences and labels, inputs are file of filenames separated by commas and labels are the names of the classes separated by commas.")]
//struct Arguments {
//   #[clap(short, long,default_value="1")]
//   threads: String,
//   #[clap(short,long)]
//   filename: String,
//}


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
