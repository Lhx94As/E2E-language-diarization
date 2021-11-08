 # Codes for paper:   
 [[Interspeech 2021] End-to-End Language Diarization for Bilingual Code-switching Speech](https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21d_interspeech.pdf)     

Pls cite as follow if you referred to this work:
@inproceedings{liu21d_interspeech,
  author={Hexin Liu and Leibny Paola García Perera and Xinyi Zhang and Justin Dauwels and Andy W.H. Khong and Sanjeev Khudanpur and Suzy J. Styles},
  title={{End-to-End Language Diarization for Bilingual Code-Switching Speech}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1489--1493},
  doi={10.21437/Interspeech.2021-82}
}
  
  Requirement:
    
  argparse  
  torch  
  tqdm  
  numpy  
  scipy
  
  * There is no script for making data, pls do it yourself and revise the code in "data_load.py" accordingly.
  * Note that the torch.cuda.deterministics=True conflicts with conv1d with dilation and this makes the code very slow, so we set it to False in train_xsa.py. Hence we don't fix the random seed in the training stage. But if you would like to do that, that's also fine, just set it to True.  
