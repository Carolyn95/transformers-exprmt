# README - on branch refactor_ner
## dataset_loaders
consolidate dataset loaders, with respect to datasets
## environments
consolidate environments with respect to package versions, only critical packages will be recorded here
## models
store all models during experimentation, but ignored in git repo
## tests
store all test cases / unit tests
## THINGS MAY CHANGE
* seed (may try different seeds to get average performance)
* eval on train, eval on test, separate from train
* eval metric
* dataset
    * ratio of class distribution(resampling)
    * data splits(randomly, stratified)
    * dataset name
* model
    * model architecture
    * pretrained models
    * training epochs
    * batch_size
    * learning_rate
    * fine tune or not 
    * ablation or not
* output path
    * containing name, training params, easier for tracking training
* command to trigger training
`python train.py --out_dir=models/test --batch=128 --lr=1e-4 --epochs=5 bert conll`
