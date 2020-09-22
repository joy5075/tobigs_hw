import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


from models import CRNN
from utils import CRNN_dataset
from tqdm import tqdm
import argparse
import os


def hyperparameters() :
    """
    argparse는 하이퍼파라미터 설정, 모델 배포 등을 위해 매우 편리한 기능을 제공합니다.
    파이썬 파일을 실행함과 동시에 사용자의 입력에 따라 변수값을 설정할 수 있게 도와줍니다.

    argparse를 공부하여, 아래에 나온 argument를 받을 수 있게 채워주세요.
    해당 변수들은 모델 구현에 사용됩니다.

    ---변수명---
    변수명에 맞춰 type, help, default value 등을 커스텀해주세요 :)
    
    또한, argparse는 숨겨진 기능이 지이이이인짜 많은데, 다양하게 사용해주시면 우수과제로 가게 됩니다 ㅎㅎ
    """
    parser = argparse.ArgumentParser(description='사용법')
#     ---path--- # 데이터셋의 위치
#     ---savepath--- # best model 저장을 위한 파일명
#     ---batch_size--- # 배치 사이즈
#     ---epochs--- # 에폭 수
#     ---optim--- # optimizer 선택
#     ---lr--- # learning rate
#     ---device--- # gpu number
#     ---img_width--- # 입력 이미지 너비
#     ---img_height--- # 입력 이미지 높이
    parser.add_argument('--path', default='./dataset', help='데이터셋의 위치')
    parser.add_argument('--savepath', default='best_model', help='best model 저장을 위한 파일명 (default: best_model)')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 사이즈 (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, help='에폭 수 (default: 30)')
    parser.add_argument('--optim', default='adam', help='optimizer 선택 (default: adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--device', type=int, default=-1, help='gpu/cpu number')
    parser.add_argument('--img_width', type=int, default=100, help='입력 이미지 너비 (default: 100)')
    parser.add_argument('--img_height', type=int, default=32, help='입력 이미지 높이 (default: 32)')

    return parser.parse_args()


def main():
    args = hyperparameters()


    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')

    # gpu or cpu 설정
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu') 

    # train dataset load
    train_dataset = CRNN_dataset(path=train_path, w=args.img_width, h=args.img_height)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # test dataset load
    test_dataset = CRNN_dataset(path=test_path, w=args.img_width, h=args.img_height)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    

    # model 정의
    model = CRNN(args.img_height, 1, 37, 256)
 
    # loss 정의
    criterion = nn.CTCLoss()
    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.999))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        assert False, "옵티마이저를 다시 입력해주세요. :("

    model = model.to(device)
    best_test_loss = 100000000
    for i in range(args.epochs):
        
        print('epochs: ', i)

        print("<----training---->")
        model.train()
        for inputs, targets in tqdm(train_dataloader):
#             ---?--- # inputs의 dimension을 (batch, channel, h, w)로 바꿔주세요. hint: pytorch tensor에 제공되는 함수 사용
            # inputs.shape 출력해본 결과 [64,1,32,100]으로 (batch, channel, h, w) 순서대로 정렬되어 있었습니다. 따라서 순서 변환없이 코드 진행하였습니다. 
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets 
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            """
            CTCLoss의 설명과 해당 로스의 input에 대해 설명해주세요.

            -> 각 문자가 어디에서 발생하는지 알 수 없으므로, 이미지에서의 ground-truth(GT) 텍스트의 모든 가능한 정렬을 시도하고 모든 점수의 합계를 취합니다. GT 텍스트의 가능성을 계산한 후 음의 로그 값을 취한 것이 CTCLoss입니다. 해당 로스의 input은 RNN의 output으로 target sequence와의 CTCLoss가 최소화되도록 학습합니다.

            """

            loss = criterion(preds, target_text, preds_length, target_length) / batch_size 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        print("<----evaluation---->")

        """
        model.train(), model.eval()의 차이에 대해 설명해주세요.
        .eval()을 하는 이유가 무엇일까요?
        
        -> model.train(): Train mode로서 학습할 때 무작위로 노드를 선택하여 선별적으로 노드를 활용함
        model.eval(): Evaluation mode로서 평가하는 과정에서 모든 노드를 사용하겠다는 의미
        
        추론을 실행하기 이전에 model.eval()을 호출하여 드롭아웃 및 배치 정규화를 평가모드로 설정해야만 추론 결과가 일관성 있게 출력됩니다. (eval 모드에서는 dropout은 비활성화, 배치 정규화는 학습에서 저장된 파라미터를 사용)

        """

        model.eval() 
        loss = 0.0

        for inputs, targets in tqdm(test_dataloader):
#             ---?---
#             ---?---
#             ---?---
#             ---?---
#             ---?---
#             ---?---
#             ---?---

            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets 
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            loss = criterion(preds, target_text, preds_length, target_length) / batch_size 
        
        print("test loss: ", loss)
        if loss < best_test_loss:
            # loss가 bset_test_loss보다 작다면 지금의 loss가 best loss가 되겠죠?
#             ---?--- 
            best_test_loss = loss
            # args.savepath을 이용하여 best model 저장하기
#             ---?---
            torch.save(model.state_dict(), args.savepath)
            print("best model 저장 성공")



if __name__=="__main__":
    main()
