## Hidden Biases in Unreliable News Detection Datasets

Official Code for the paper:

"**Hidden Biases in Unreliable News Detection Datasets**"

Xiang Zhou, Heba Elfardy, Christos Christodoulopoulos, Thomas Butler and Mohit Bansal

EACL 2021

### Dependencies
The code is tested on Python 3.7 and PyTorch 1.6.0

Other dependencies are listed in `requirements.txt` and can be installed by running `pip install -r requirements.txt` 

### Datasets
##### Download Original Datasets

The experiments and results in our paper mainly involve two datasets: NELA and FakeNewsNet

For the NELA dataset, we use both the [2018 version](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ULHLCB) and the [2019 version](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O7FWPO). To reproduce experiments, please first download both versions (on the download page, please select all and choose the original format) and put it under the `data` directory. Then, decompress `nela/2018/articles.tar.gz` and `nela/2019/nela-gt-2019-json.tar.bz2` and put them under the original directory. The structure of `data` should look like this:

```
data
└── nela
    ├── 2018
    │   ├── articles
    │   │   └── ... 
    │   ├── articles.db.gz
    │   ├── articles.tar.gz
    │   ├── labels.csv
    │   ├── labels.txt
    │   ├── nela_gt_2018-new_schema.tar.bz2
    │   ├── README.md
    │   └── titles.tar.gz
    └── 2019
        ├── labels.csv
        ├── nela-eng-2019
        │   └── ... 
        ├── nela-gt-2019-json.tar.bz2
        ├── nela-gt-2019.tar.bz2
        ├── README-1.md
        ├── README.md
        └── source-metadata.json
```

The FakeNewsNet dataset can be crawled using the code from [its official GitHub repo](https://github.com/KaiDMML/FakeNewsNet). After downloading the dataset, please put it also under the `data/fakenewsnet_dataset/raw`, and the whole `data` folder should look like this:

```
data
├── fakenewsnet_dataset
│   └── raw
└── nela
    └── ...
```

The default location of `data` directory is under the root directory. If you prefer storing your data in other locations, you can change the variables in `constants.py`


##### Create Dataset Splits
To create the random/site/time split of NELA in the paper, run `python data_helper.py nela {site, time, random}` 

To create the random label split, run `python data_helper.py nela random_label` (Note you have to manually rename the split dataset after creating the random label split)

To create the split of FakeNewsNet in the paper, run `python data_helper.py fnn {site, time, random}`



### Train Baseline Models
Example scripts to train baseline models used in this paper can be found under the `scripts` directory (Please refer to Sec. 4.1 in the [paper](paper_link) for detailed descriptions of the baseline models). You can change the dataset path to train different baselines.

To train the logistic regression baseline, run `bash scripts/lr.sh`

To train the title-only RoBERTa models, run `bash scripts/roberta_title.sh`

To train the title+article RoBERTa models, run `bash scripts/roberta_title_article.sh`

### Reproducing Analysis Experiments

##### Source Level Results
1. Get the predictions on the validation set (by running eval commands in the model training scripts).
2. To get source-level accuracies, run `python source_evaluation.py --pred_file [PREDICTION_FILE] --key_file [KEY_FILE] --pred_type [PRED_TYPE]`. Set `PRED_TYPE` to `clean` for the logistic regression model and the title-only RoBERTa model and `full` for the title+article RoBERTa model due to different output formats. Please refer to the python file for the details of other arguments.

##### Extracting Salient Features
1. Train a logistic regression model using `bash scripts/lr.sh` and save the trained model by adding the `save_model [MODEL_PATH]` argument.
2. To extract salient features from logistic regression baselines, run `python analysis_lr.py --model_path [MODEL_PATH]`. Please refer to the python file for the details of other arguments.

##### Site Similarity Analysis
1. Create 5 different domain splits using different seeds by running `python data_helper.py nela site [SEED]`
2. To get site similarity results in Table 7 in the paper, train 5 title+article baselines on each of these 5 different domain splits by running `bash scripts/roberta_title_article.sh` on and put all the predictions under the `output` directory. Change the `SAVE_DIRS` and the `SITE_PREDS` variables in `site_similarity.py` to match your saved path and run `python site_similarity.py`

##### Word Cloud Visualization
1. Save the titles with correct or wrong predictions in file `correct.title` and `wrong.title` respectively by running `python dump_titles.py --pred_file [PREDICTION_FILE] --key_file [KEY_FILE] --pred_type [PRED_TYPE]`. Set `PRED_TYPE` to `clean` for the logistic regression model and the title-only RoBERTa model and `full` for the title+article RoBERTa model due to different output formats. Then, put `correct.title` and `wrong.title`  in the same directory as `draw_cloud_unigram.py`.
2. To draw the word cloud showing the most salient words in examples with correct or wrong (determined by the PRINT_TYPE variable in the script) prediction, run `python draw_cloud_unigram.py`


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

