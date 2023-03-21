from os.path import splitext
def get_label(im):
    return splitext(get_labels(im))[0] + '.txt'

def get_labels(ims):
    return "/labels/".join(ims.rsplit("/images/", 1))

from argparse import ArgumentParser
def parse_opt():
    parser = ArgumentParser()
    parser.add_argument("--max-bg-ratio", type=float, default=.999)
    parser.add_argument('old_yaml')
    parser.add_argument('new_yaml')
    return parser.parse_args()


from os import listdir, makedirs, symlink
from os.path import join, abspath, isfile
from random import shuffle
from yaml import safe_load as load
def run(old_yaml, new_yaml, max_bg_ratio):
    old = load(open(old_yaml))
    new = load(open(new_yaml))
    l_n = {l:n for n, l in new['names'].items()}
    old_cls_new = {str(o): str(l_n[l]) for o, l in old['names'].items()
                   if l in l_n}
    splits = ['val', 'train']
    if 'test' in new:
        splits.append('test')
    for split in splits:
        fg = 0
        background = []
        for d in new, old:
            d[split] = join(d.get('path', ''), d[split])
        makedirs(new[split])
        makedirs(get_labels(new[split]))
        for imf in listdir(old[split]):
            oldim = join(old[split], imf)
            newim = join(new[split], imf)
            labels = [" ".join([old_cls_new[parts[0]], parts[1]])
                      for label in open(oldlb).readlines()
                      if (parts := label.split(" ", 1))[0] in old_cls_new
                      ] if isfile(oldlb := get_label(oldim)) else []
            if not labels:
                background.append((oldim, newim))
            else:
                fg += 1
                symlink(abspath(oldim), newim)
                open(get_label(newim), 'w').writelines(labels)

        shuffle(background)
        background = background[:int(max_bg_ratio * fg / (1 - max_bg_ratio))]
        for oldim, newim in background:
            symlink(abspath(oldim), newim)
