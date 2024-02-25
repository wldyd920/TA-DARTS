# GraphViz error solution
# conda install -c anaconda python-graphviz
# conda install -c anaconda pydot

import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='green')
    g.node("c_{k-1}", fillcolor='green')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='cyan')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='red')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=False)


for i in [0, 9, 19, 29, 39, 49]:
    genotype_name = f'TADARTS45_epoch{i}'
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, 'normal_'+genotype_name)
    plot(genotype.reduce, 'reduce_'+genotype_name)


# for i in [0, 9, 19, 29, 39, 49]:
#     genotype_name = f'TADARTS45_epoch{i}'
#     try:
#         genotype = eval('genotypes.{}'.format(genotype_name))
#     except AttributeError:
#         print("{} is not specified in genotypes.py".format(genotype_name))
#         sys.exit(1)

#     plot(genotype.normal, 'normal_'+genotype_name)
#     plot(genotype.reduce, 'reduce_'+genotype_name)
