# MFF-code-generation
## Neural Code Generation Based on Multi-Modal Fine-Grained Feature Fusion
This project refers to neural code generation based on multi-modal fine-grained feature fusion. Its main idea is as follows: first, the code fragment is characterized as three modalities, namely token sequences (Tokens), abstract syntax trees (ASTs), and API dependency graphs (ADGs); secondly, the node-level information of AST and ADG of the code are matched with the information the token of the code fragment; thirdly, the node-level information of AST and ADG is then fused with the token of the code fragment at a fine-grained level; Finally, the fused information and the corresponding natural language descriptions are put into the transformer for training.
## Requirements
  * Pytorch 1.8.0
  * Python 3.6.5
  * Network 2.3
  * Numpy 1.19.5
  * Nltk 3.6.2
  * Pandas 1.1.5
  * Tensorflow 1.15.0
  * bert-serving-client 1.10.0
  * bert-serving-server 1.10.0
## Usage
###To train a new model and to predict the code (HS)
```Python3 Model/HS_Trans_tok_AST.py```
###To train three new model and to predict the code (MTG)
```Python3 Mode/MG_Trans_tok_ADG.py```
```Python3 Mode/MG_Trans_tok_AST.py```
```Python3 Mode/MG_Trans_tok_AST_ADG.py```
