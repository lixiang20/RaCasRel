## A Relation Aware Embedding Mechanism for Relation Extraction 
some of our codes are from https://github.com/weizhepei/CasRel

## Requirements

This repo was tested on Python 3.6 and Keras 2.2.4. The main requirements are:

- tqdm
- keras-bert==0.82.0
- tensorflow-gpu == 1.13.1

## Datasets

- [NYT](https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3)
- [WebNLG](https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj)

1. **Get datasets**

    Download the two datasets above. Then decompress it under `data/NYT/` or `data/WebNLG/`.


## Usage

1. **Get pre-trained model BERT**

   Download Google's pre-trained BERT model **[(`BERT-Base, Cased`)](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**. Then decompress it under `pretrained_bert_models/`.

2. **Train and select the model**

   Specify the running mode and dataset at the command line

   ```shell
   python run.py --train=True --dataset=NYT
   ```

   The model weights that lead to the best performance on validation set will be stored in `saved_weights/NYT/` or `saved_weights/WebNLG/`.
   
## Baselines
- [NovelTagging](https://www.aclweb.org/anthology/P17-1113.pdf)
- [CopyRE](https://www.aclweb.org/anthology/P18-1047.pdf)
- [OrderRE](https://www.aclweb.org/anthology/D19-1035.pdf)
- [CopyMTL](https://ojs.aaai.org/index.php/AAAI/article/view/6495)
- [RSAN](https://researchmgt.monash.edu/ws/portalfiles/portal/320133657/319920316_oa.pdf)
- [ETL-Span](https://ecai2020.eu/papers/615_paper.pdf)
- [WDec](https://ojs.aaai.org/index.php/AAAI/article/view/6374)
- [SMHSA](https://www.ijcai.org/Proceedings/2020/0524.pdf)
- [GraphRel](https://www.aclweb.org/anthology/P19-1136.pdf)
- [CasRel](https://www.aclweb.org/anthology/2020.acl-main.136/)

