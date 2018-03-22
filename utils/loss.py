import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(object):
    def __init__(self, margin=0.5):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, dist_ap, dist_an):
        """
        :param dist_ap: distance of anchor and positive [N]
        :param dist_an: distance of anchor and negative [N]
        :return: loss [1]
        """

        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


def global_euclidean_dist(x, y):

    """
    :param x: (pytorch Variable) global features [M, d]
    :param y: (pytorch Variable) global features [N, d]
    :return: pytorch Variable euclidean distance matrix [M, N]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist_mat = xx + yy
    dist_mat.addmm_(1, -2, x, y.t())
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist_mat


def local_euclidean_dist(x, y):
    """
    :param x: (pytorch Variable) local features [M, m, d]
    :param y: (pytorch Variable) local features [N, n, d]
    :return: pytorch Variable euclidean distance matrix [M, N]
    """
    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    dist_mat = global_euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    dist_mat = shortest_dist(dist_mat)
    return dist_mat


def shortest_dist(dist_mat):
    """
    :param dist_mat: distance matrix [m, n, *]
    :return: shortest distance [*]
    """
    m, n = dist_mat.size()[:2]
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist


def hard_example_mining(dist_mat, labels):
    """
    :param dist_mat: (pytorch Variable) distance matrix [N, N]
    :param labels: (pytorch LongTensor) labels [N]
    :return: dist_ap, dist_an [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    N = dist_mat.size(0)

    pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, _ = torch.max(dist_mat[pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat[neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def triplet_loss(tri_loss, feat, labels):
    """
    :param tri_loss: TripletLoss object margin=0.5
    :param feat: (pytorch Variable) features [N, d]
    :param labels: (pytorch LongTensor) labels [N]
    :return: loss
    """
    dist_mat = global_euclidean_dist(feat, feat)
    dist_ap, dist_an = hard_example_mining(dist_mat, labels)
    loss = tri_loss(dist_ap, dist_an)

    return loss


def mixed_loss(tri_loss, global_feat, local_feat, labels, mutual_feature=False):
    """
    :param tri_loss: TripletLoss object margin=1
    :param global_feat: (pytorch Variable) global features [N, d]
    :param local_feat: (pytorch Variable) local features [N, m, d]
    :param labels: (pytorch LongTensor) labels [N]
    :return: loss
    """
    global_dist_mat = global_euclidean_dist(global_feat, global_feat)
    global_dist_ap, global_dist_an = hard_example_mining(global_dist_mat, labels)
    local_dist_mat = local_euclidean_dist(local_feat, local_feat)
    local_dist_ap, local_dist_an = hard_example_mining(local_dist_mat, labels)
    loss = tri_loss(global_dist_ap+local_dist_ap, global_dist_an+local_dist_an)

    if mutual_feature:
        return loss, global_dist_mat

    return loss
