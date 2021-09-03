# MFF-code-generation
## Neural Code Generation Based on Multi-Modal Fine-Grained Feature Fusion
This project refers to neural code generation based on multi-modal fine-grained feature fusion. Its main idea is as follows: first, the code fragment is characterized as three modalities, namely token sequences (Tokens), abstract syntax trees (ASTs), and API dependency graphs (ADGs); secondly, the node-level information of AST and ADG of the code are matched with the information the token of the code fragment; thirdly, the node-level information of AST and ADG is then fused with the token of the code fragment at a fine-grained level; Finally, the fused information and the corresponding natural language descriptions are put into the transformer for training.
## Requirements
  *Pytorch 1.8.0
  *Python 3.6.5
  *Network 2.3
