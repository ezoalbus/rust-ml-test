# README

[Qiitaの記事](https://qiita.com/ezoalbus/items/f5eaea32404098e6ca37)

## 環境
- Ubuntu 20.04 LTS on WSL2 (Windows 10)

## クイックスタート
アルゴリズムはナイーブベイズ（二値分類）、データセットはlinfa-datasetsに含まれている[UCIのワインデータ](https://archive.ics.uci.edu/ml/datasets/wine+quality)を使う。

### Cargo.toml

`cargo new <project_name>`でプロジェクトを作成し、生成されたCargo.tomlファイルに下記のように追記する。なお、linfaの"features"においてopenblasを指定しているが、WindowsとmacOSはBLASのバックエンドがIntel MKLのみらしい（Intel MKLをバックエンドで使う場合は、[この記事](https://qiita.com/termoshtt/items/236cec0e10a0ff37a97f)が参考になる？ ... 未確認）。

``` toml: Cargo.toml
[package]
name = "rust-bayes"
version = "0.1.0"
edition = "2021"

 # See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
linfa = { version = "0.5.0", features=["openblas-system"] }
linfa-datasets = { version = "0.5.0", features = ["winequality"] }
linfa-bayes = { version = "0.5.0" }

```

### main.rs

公式の[example](https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-bayes/examples/winequality.rs)からほぼそのまま拝借した。

``` rust: main.rs
use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{GaussianNb, Result};


fn main() -> Result<()> {
    // データセットを読み込み、ターゲット（ワインの品質の評価値）を二値に変換
    //     品質は、0から10の間で評価されており、0が最低で10が最高
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);

    // モデルの訓練
    let model = GaussianNb::params().fit(&train)?;

    // 推論
    let pred = model.predict(&valid);

    // 混同行列の計算
    let cm = pred.confusion_matrix(&valid)?;

    // 混同行列と精度の出力（MCCはマシューズ相関係数）
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
    
    Ok(())
}
```

### 実行

``` sh
~/rust/rust-bayes$ cargo run
   Compiling rust-bayes v0.1.0 (/home/ezoalbus/rust/rust-bayes)
    Finished dev [unoptimized + debuginfo] target(s) in 2.10s
     Running `target/debug/rust-bayes`

classes    | good       | bad       
good       | 10         | 7         
bad        | 12         | 130       

accuracy 0.8805031, MCC 0.45080975
```

途中、"error: linking with \`cc\` failed: exit status: 1" が出たので、`sudo apt install libopenblas-dev`などをした。

---
おわり

## 参考
- https://blog.logrocket.com/machine-learning-in-rust-using-linfa/
- https://www.wasm.builders/ajay272191/standard-k-mean-clustering-on-random-data-in-rustwasm-using-linfa-rust-14bi
