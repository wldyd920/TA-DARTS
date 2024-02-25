import  os,sys,glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn

from    model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=36, help='batch size')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
# parser.add_argument('--exp_path', type=str, default='exp/model.pt', help='path of pretrained model')
parser.add_argument('--exp_path', type=str, default='exp/cifar10-20220627-171615/trained.pt', help='path of pretrained model')
# 34: cifar10-20220517-172413
# 35: cifar10-20220523-113602
# 53:
# 54: cifar10-20220711-135946
# 59: cifar10-20220717-221821

# 44: cifar10-20220630-171421
# 45: cifar10-20220627-171414
# 46: cifar10-20220627-171615
# 47: cifar10-20220627-171923
# 48: cifar10-20220701-145142

# 49: cifar10-20220711-144536
# 50: cifar10-20220711-144600
# 51: cifar10-20220711-144624
# 52: cifar10-20220711-144211
# 55: cifar10-20220717-221811
# 56: cifar10-20220717-221816

parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=30, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS46', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')



def main():


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # equal to: genotype = genotypes.DARTS_v2
    genotype = eval("genotypes.%s" % args.arch)
    print('Load genotype:', genotype)
    model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype).cuda()
    utils.load(model, args.exp_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()

    _, test_transform = utils._data_transforms_cifar10(args)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():

        for step, (x, target) in enumerate(test_queue):

            x, target = x.cuda(), target.cuda(non_blocking=True)

            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            batchsz = x.size(0)
            objs.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
