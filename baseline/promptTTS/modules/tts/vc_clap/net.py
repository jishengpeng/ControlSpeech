import torch
from torch import nn
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
    
class SpeakerClassifier(nn.Module):
    def __init__(self,input_dim=256,output_dim=20):
        super(SpeakerClassifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim,input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim*2, output_dim),
            nn.ReLU(inplace=True)
        )
 
    def forward(self,x):
        # x: [b,C]
        x = self.classifier(x) 
        # return nn.functional.softmax(x,dim=1) 
        return x
 
    def compute_accuracy(self,output,labels):
        preds = torch.argmax(output,dim=1) # 获取每个样本的预测标签
        correct = torch.sum(preds == labels).item() # 计算正确预测的数量
        accuracy = correct / len(labels) # 除以总样本数得到准确率
        return accuracy