import pickle as pkl
import sys

from sklearn.metrics.pairwise import pairwise_kernels

from semantic import *
from utils import normalize


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# dataset == 'citeseer'...
def process_data(dataset):
    names = ['y', 'ty', 'ally', 'x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/cache/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty, ally, x, tx, allx, graph = tuple(objects)
    print(graph)
    test_idx_reorder = parse_index_file("../data/cache/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    print(features)
    f = open('../data/{}/{}.adj'.format(dataset, dataset), 'w+')
    for i in range(len(graph)):
        adj_list = graph[i]
        for adj in adj_list:
            f.write(str(i) + '\t' + str(adj) + '\n')
    f.close()

    label_list = []
    for i in labels:
        label = np.where(i == np.max(i))[0][0]
        label_list.append(label)
    np.savetxt('../data/{}/{}.label'.format(dataset, dataset), np.array(label_list), fmt='%d')
    np.savetxt('../data/{}/{}.test'.format(dataset, dataset), np.array(test_idx_range), fmt='%d')
    np.savetxt('../data/{}/{}.feature'.format(dataset, dataset), features, fmt='%f')


''' process cora/citeseer/pubmed data '''


# process_data('citeseer')

def construct_graph(dataset, features, topk, fn):
    fname = '../data/' + dataset + '/knn' + str(fn) + '/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    if fn == 1:
        # Gaussian
        dist = pairwise_kernels(features, metric='rbf', gamma=0.5)

    elif fn == 2:
        # Cosine
        dist = pairwise_kernels(features, metric='cosine')

    elif fn == 3:
        # Sigmoid
        dist = pairwise_kernels(features, metric='sigmoid')

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset, f):
    for topk in range(2, 10):
        data = np.loadtxt('../data/' + dataset + '/' + dataset + '.feature', dtype=float)
        # print(data)
        construct_graph(dataset, data, topk, f)
        f1 = open('../data/' + dataset + '/knn' + str(f) + '/tmp.txt', 'r')
        f2 = open('../data/' + dataset + '/knn' + str(f) + '/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()


'''generate KNN graph'''


def construct_ppmi(dataset):
    fname = '../data/' + dataset + '/ppmi.npz'
    print(fname)
    data = '../data/' + dataset + '/' + dataset + '.edge'

    struct_edges = np.genfromtxt(data, dtype=np.int32)  #
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(struct_edges.max() + 1, struct_edges.max() + 1), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
    ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(nsadj, path_len=2, k=2.0)
    # ppmi = diffusion_fun_improved(nsadj, path_len=2)

    sparse.save_npz(fname, ppmi, True)


construct_ppmi('Citeseer')
'''generate ppmi'''
