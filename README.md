 # Codes for paper:   
 # End-to-End Language Diarization for Bilingual Code-switching Speech  
 Accepted to Interspeech 2021, will be held in the end of Aug.
  
  Requirement:
    
  argparse  
  torch  
  tqdm  
  numpy  
  scipy
  
  * There is no script for making data, pls do it yourself and revise the code in "data_load.py" accordingly.
  * Note that the torch.cuda.deterministics=True conflicts with conv1d with dilation and this makes the code very slow, so we set it to False in train_xsa.py
