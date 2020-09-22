# from torch.utils.data import ---?---
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import collections
from glob import glob
import os
from PIL import Image

"""
main.py 함수를 참고하여 다음을 생각해봅시다.

1. CRNN_dataset은 어떤 모듈을 상속받아야 할까요?
    -> torch.utils.data.Dataset
    
2. CRNN_dataset의 역할은 무엇일까요? 왜 필요할까요?
    -> torch.utils.data.Dataset은 데이터셋을 나타내는 추상클래스로, 사용하고자하는 데이터셋에 상속하고 오버라이드를 해야합니다. 이를 통해 custom dataset을 만들 수 있습니다. 
    
3. 1.의 모듈을 상속받는다면 __init__, __len__, __getitem__을 필수로 구현해야 합니다. 각 함수의 역할을 설명해주세요.
    1)__init__: 데이터셋의 전처리를 해주는 부분
    2)__len__: 데이터셋의 길이. 
    3)__getitem__: 데이터셋에서 특정 1개의 샘플을 가져오는 함수

"""


# class CRNN_dataset(---?---):
class CRNN_dataset(Dataset):
    def __init__(self, path, w=100, h=32, alphabet='0123456789abcdefghijklmnopqrstuvwxyz', max_len=36):
        self.max_len=max_len
        self.path = path
        self.files = glob(path+'/*.jpg') 
        self.n_image = len(self.files)
        assert (self.n_image > 0), "해당 경로에 파일이 없습니다. :)"

        self.transform = transforms.Compose([
#             ---?---, # image 사이즈를 w, h를 활용하여 바꿔주세요.
#             ---?--- # tensor로 변환해주세요.
            transforms.Resize((h, w)), # If size is a sequence like (h, w), output size will be matched to this
            transforms.ToTensor()
        ])
        """
        strLabelConverter의 역할을 설명해주세요.
        1. text 문제를 풀기 위해 해당 함수는 어떤 역할을 하고 있을까요?

        -> 구글링 검색 결과, Convert between str and label , Insert 'blank' to the alphabet for CTC 로 나오는데, 이해한 바로는, 예를 들어 'to'란 단어가 쓰여진 이미지가 있을 때, o가 t보다 많은 범위를 차지할 경우 'ttooo'란 단어를 얻게 될 수 있는데, 중복된 't'와 'o'를 제거하기 위해서 blank(- 또는 pseudo-character)를 넣어주고 ground-truth 텍스트만을 CTC loss에 제공해주는 함수인 것 같습니다.

        2. encode, decode의 역할 설명

        - encode: 위에서 언급했듯이 중복된 문자 사이에 blank를 넣어주는 역할
        - decode: blank들을 제거해주는 역할 

        """
        self.converter = strLabelConverter(alphabet) 
        
    def __len__(self):
#         return ---?--- # hint: __init__에 정의한 변수 중 하나
        return self.n_image

    def __getitem__(self,idx):
        label = self.files[idx].split('_')[1] # 파일명이 1_string으로 되어있음. 즉 해당 이미지의 문자가 label이 되는 셈
        img = Image.open(self.files[idx]).convert('L')
        img = self.transform(img) # Resize(h,w) & ToTensor
        """
        max_len이 왜 필요할까요? # hint: text data라는 점

        -> alphabet에 해당하는 0123456789abcdefghijklmnopqrstuvwxyz가 36개인데, label_text가 고정된 길이(36)로 출력이 되게끔 하려고 쓰는 것 같습니다.

        """

        if len(label) > self.max_len:
#             label = label[:self.mfax_len] # 오타..? 
            label = label[:self.max_len]  
        label_text, label_length = self.converter.encode(label) # insert blank

        if len(label_text) < self.max_len:
            temp = torch.ones(self.max_len-len(label), dtype=torch.int)
            label_text = torch.cat([label_text, temp])

#         return ---?---, (---?---, ---?---) # hint: main.py를 보면 알 수 있어요 :)
        return img, (label_text, label_length) 


# 아래 함수는 건드리지 마시고, 그냥 쓰세요 :)
class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-' 

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
