# nmt4code
Repository for neural machine translation 4 code:
Generate AST from binary [contains code for neural machine translation part]

Note: v1 dataset in the paper corresponds to v2 in the code and v2 dataset in the paper corresponds to v3 in the code.

Steps to reproduce the results:
1. sample new dataset using code_gen or you can use our dataset
2. prepare data for NMT using experiments/prepare_data{v2/v3}.sh
	- This will generate pickle files for training and testing
	- you can use provided pickle files for training, validation, and test
	[][]
3. Our code has implementation of following papers for AST generation
	- Sequence to Sequence Learning with Neural Networks [][https://arxiv.org/pdf/1409.3215.pdf]
	- Neural Machine Translation By Jointly Learning to Align and Translate [][https://arxiv.org/pdf/1409.0473.pdf]
	- Attention Is All You Need (Transformer) [][https://arxiv.org/pdf/1706.03762.pdf]
	- Tree-to-tree Neural Networks for Program Translation [][https://arxiv.org/pdf/1802.03691.pdf]
	- Learning Deep Generative Models of Graphs [][https://arxiv.org/pdf/1803.03324.pdf] [Modified version for AST generation in BFS]

Link to the presentation slides:
[][https://docs.google.com/presentation/d/1dgv1_KMehkBmrbbt3veWRbdc3qFjy8Ki2No8iBa6GuQ/edit?usp=sharing]

special thanks to [][https://github.com/bentrevett/pytorch-seq2seq]

If you use code from this repo, please cite our paper:
	@inproceedings{dharma2023neural,
	  title={Neural Machine Translation for Recovering ASTs from Binaries},
	  author={Dharma, KC and Ferra, Tito and Morrison, Clayton T},
	  booktitle={2023 IEEE 3rd International Conference on Software Engineering and Artificial Intelligence (SEAI)},
	  pages={80--85},
	  year={2023},
	  organization={IEEE}
	}

