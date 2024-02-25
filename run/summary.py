import torch
from torchsummary import summary
from model import NetworkCIFAR as Network
import  argparse
import  genotypes

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--arch', type=str, default='DARTS43', help='which architecture to use')
args = parser.parse_args()

genotype = eval("genotypes.%s" % args.arch)

model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype)

model_path = "C:/Users/user/VSC/DARTS/hist/43/cifar10-20220623-163746/trained.pt"
mymodel = model.load_state_dict(torch.load(model_path))
model.eval()

input_parameter = next(model.parameters())
inputsize = input_parameter.size()
print("input size : ", inputsize)
# summary(mymodel, input_size=(args.init_ch, 32, 32))

# summary(mymodel, input_size=(args.init_ch, H, W))

# model.load_state_dict(torch.load(model_path))
# model.eval()