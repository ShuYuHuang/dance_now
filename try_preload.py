import sys,os
from pathlib import Path
sys.path.append(str(Path("./")))
import inference_openpose as inf



while (1):
    input()
    inf.transfer("../test/test_img/test_img.png","malcom")
  