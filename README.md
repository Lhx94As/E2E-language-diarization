 # Codes for paper:   
 [[Interspeech 2021] End-to-End Language Diarization for Bilingual Code-switching Speech](https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21d_interspeech.pdf)     

  
  Requirement:
    
  argparse  
  torch  
  tqdm  
  numpy  
  scipy
  
  * There is no script for making data, pls do it yourself and revise the code in "data_load.py" accordingly.
  * Note that the torch.cuda.deterministics=True conflicts with conv1d with dilation and this makes the code very slow, so we set it to False in train_xsa.py
