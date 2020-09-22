# models.py 파일과 논문을 바탕으로 빈칸을 채워주세요.
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # convRelu(0) 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, stride=1, paddidng=1), # convRelu(1)
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # convRelu(2, True)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # convRelu(3) 
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), # convRelu(4, True)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), # convRelu(5)
            nn.ReLU(),
            nn.Conv2d(512, 512, 2, stride=1, padding=0), # convRelu(6, True)
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
            
        self.rnn_model = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True),
            nn.LSTM(256, 256, bidirectional=True)
        )


#         self.embedding = nn.Linear(---?---, 37)
        self.embedding = nn.Linear(256*2, 37)

    def forward(self, input):
#         conv = ---?---
        conv = self.cnn(input)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
#         output, _ = ---?---
        output, _ = self.rnn(conv)
        seq_len, batch, h_2 =  output.size()
        output = output.view(seq_len * batch, h_2)
#         output = ---?---
        output = self.embedding(output)
        output = output.view(seq_len, batch, -1)
        return output

