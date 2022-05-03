use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{GaussianNb, Result};


fn main() -> Result<()> {
    // データセットを読み込み、ターゲット（ワインの品質の評価値）を二値に変換
    //     品質は、0から10の間で評価されており、0が最低10が最高
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
