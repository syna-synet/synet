from os.path import join, basename

aP_curve_points = 10000

from argparse import ArgumentParser, Namespace
def parse_opt() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("data_yamls", nargs="+")
    parser.add_argument("--out-dirs", nargs="+")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--print-jobs", action="store_true")
    parser.add_argument("--precisions", nargs="+", type=float, required=True,
                        help="CANNOT BE SPECIFIED WITH --precisions=...' "
                        "SYNTAX: MUST BE '--precisions PREC1 PREC2 ...'")
    return parser.parse_known_args()[0]

from yolov5.utils.general import xywh2xyxy
from torch import tensor
from numpy import genfromtxt
def txt2xyxy(txt : str) -> tensor:
    """Convert txt path to array of (cls, x1, y1, x2, y2[, conf])"""
    a = tensor(genfromtxt(txt, ndmin=2))
    a[:, 1:5] = xywh2xyxy(a[:, 1:5])
    return a

from os.path import isfile, isdir, isabs
from yolov5.utils.dataloaders import check_dataset, img2label_paths
from glob import glob
def get_gt(data_yaml : str) -> dict:
    """Obtain {"###.txt" : (Mx5 array)} dictionary mapping each data
sample to an array of ground truths.  See get_pred().

    """
    path = check_dataset(data_yaml)['val']
    f = []
    for p in path if isinstance(path, list) else [path]:
        if isdir(p):
            f += glob(join(p, "**", "*.*"), recursive=True)
        else:
            f += [t if isabs(t) else join(dirname(p), t)
                  for t in open(p).read().splitlines()]
    return {basename(l): txt2xyxy(l)
            for l in img2label_paths(f)}

from os import listdir
def get_pred(pred : str) -> dict:
    """from model output dir from validation, pred, obtain {"###.txt"
: (Mx6 array)} mapping each data sample to array of predictions (with
confidence) on that sample.  See get_gt().

    """
    return {name : txt2xyxy(join(pred, name))
            for name in listdir(pred)}

from torch import tensor, stack, cat
iouv=tensor([.5])
from yolov5.val import process_batch
def get_tp_ngt(gt : dict, pred : dict) -> tuple:
    """From ground truth and prediction dictionaries (as given by
get_gt() and get_pred() funcs resp.), generate a single Mx3 array, tp,
for the entire dataset, as well as a dictionary, gt = { (class : int)
: (count : int) }, giving the count of each ground truth.  The array
is interpreted as meaning there are M predictions denoted by (conf,
class, TP) giving the network predicted confidence, the network
predicted class, and a flag TP which is 1 if the sample is considered
a true positive.

    """
    # after this point, we don't care about which pred came from which
    # data sample in this data split
    tp = cat([stack((pred[fname][:, 5], # conf
                     pred[fname][:, 0], # class
                     process_batch(pred[fname][:,[1,2,3,4,5,0]],
                                   gt[fname],
                                   iouv
                                   ).squeeze(1)), # TP
                    -1)
              for fname in pred])
    l = cat([gt[fname][:, 0] for fname in pred])
    ngt = {int(c.item()) : (l == c).sum() for c in l.unique()}
    return tp, ngt

from torch import cumsum, arange, linspace
from numpy import interp
from matplotlib.pyplot import (plot, legend, title, xlabel, ylabel,
                               savefig, clf, scatter, grid, xlim, ylim)
def get_aps(tp : tensor, ngt : dict, precisions : list, label : str,
            project : str, glob_confs : [list, None] = None) -> list:
    """This is the main metrics AND plotting function.  All other
functions exist to "wrangle" the data into an optimal format for this
function. From a 'tp' tensor and 'ngt' dict (see get_tp_ngt()),
compute various metrics, including the operating point at
'precisions'[c] for each class c.  Plots are labeled and nammed based
on 'label', and placed in the output dir 'project'.  Additionally, if
glob_confs is also given, plot the operating point at that confidence
threshold.  Returns the confidence threshold corresponding to each
precision threshold in 'precisions'.

    """
    # if there are fewer precision thresholds specified than classes
    # present, and only one precision is specified, use that precision
    # for all classes
    if max(ngt) > len(precisions) - 1:
        if len(precisions) == 1:
            print("applying same precision to all classes")
            precisions *= max(ngt)
        else:
            print("specified", len(precisions), "precisions, but have",
                  max(ngt)+1, "classes")
            exit()
    # Main loop.  One for each class.  AP calculated at the end
    AP, confs, op_P, op_R, half_P, half_R = [], [], [], [], [], []
    if glob_confs is not None: glob_P, glob_R = [], []
    for cls, prec in enumerate(precisions):
        print("For class:", cls)

        # choose class and omit class field
        selected = tp[tp[:,1] == cls][:,::2]

        # sort descending
        selected = selected[selected[:, 0].argsort(descending=True)]

        # calculate PR values
        assert len(selected.shape) == 2
        tpcount = cumsum(selected[:,1], 0).numpy()
        P = tpcount / arange(1, len(tpcount) + 1)
        R = tpcount / ngt.get(cls, 0)
        # enforce that P should be monotone
        P = P.flip(0).cummax(0)[0].flip(0)

        # calculate operating point from precision.
        # operating index is where the precision last surpasses precision thld
        # argmax on bool array returns first time condition is met.
        # Precision is not monotone, so need to reverse, argmax, then find ind
        assert len(P.shape) == 1
        confs.append(selected[(P < prec).byte().argmax() -1, 0])
        op_ind = (selected[:,0] <= confs[-1]).byte().argmax() - 1
        op_P.append(P[op_ind])
        op_R.append(R[op_ind])
        print(f"Conf, Precision, Recall at operating point precision={prec}")
        print(f"{confs[-1]:.6f}, {op_P[-1]:.6f}, {op_R[-1]:.6f}")

        if glob_confs is not None:
            # if glob threshold is passed, also find that PR point
            glob_ind = (selected[:,0] <= glob_confs[cls]).byte().argmax() - 1
            glob_P.append(P[glob_ind])
            glob_R.append(R[glob_ind])
            print("Conf, Precision, Recall at global operating point:")
            print(f"""{glob_confs[cls]:.6f}, {glob_P[-1]
                      :.6f}, {glob_R[-1]:.6f}""")

        # show .5 conf operating point
        half_ind = (selected[:,0] <= .5).byte().argmax() - 1
        half_P.append(P[half_ind])
        half_R.append(R[half_ind])
        print(f"Conf, Precision, Recall at C=.5 point")
        print(f"{.5:.6f}, {half_P[-1]:.6f}, {half_R[-1]:.6f}")

        # generate plotting points/AP calc points
        Ri = linspace(0, 1, aP_curve_points)
        Pi = interp(Ri, R, P)
        # use these values for AP calc over raw to avoid machine error
        AP.append(Pi.sum() / aP_curve_points)
        print("class AP:", AP[-1].item(), end="\n\n")
        plot(Ri, Pi, label=f"{cls}: AP={AP[-1]:.6f}")

    # calculate mAP
    mAP = sum(AP)/len(AP)
    print("mAP:", mAP, end="\n\n\n")
    title(f"{basename(label)} mAP={mAP:.6f}")

    # plot other points
    scatter(op_R, op_P, label="precision operating point")
    scatter(half_R, half_P, label=".5 conf")
    if glob_confs is not None:
        scatter(glob_R, glob_P, label="global operating point")

    # save plot
    legend()
    xlabel("Recall")
    ylabel("Precision")
    grid()
    xlim(0, 1)
    ylim(0, 1)
    savefig(join(project, f"{basename(label)}.png"))
    clf()

    return confs

def metrics(data_yamls : list, out_dirs : list, precisions : list,
            project : str) -> None:
    """High level function for computing metrics and generating plots
for the combined data plus each data split.  Requires list of data
yamls, data_yamls, model output dirs, out_dirs, classwise precision
thresholds, precisions, and output dir, project.

    """
    tp_ngt = {}
    for data_yaml, out_dir in zip(data_yamls, out_dirs):
        tp_ngt[data_yaml] = get_tp_ngt(get_gt(data_yaml),
                                       get_pred(join(out_dir, 'labels')))
    print("Done reading results.  Results across all data yamls:", end="\n\n")
    confs = get_aps(cat([tp for tp, _ in tp_ngt.values()]),
                    {c : sum(ngt.get(c, 0) for _, ngt in tp_ngt.values())
                     for c in set.union(*(set(ngt.keys())
                                          for _, ngt in tp_ngt.values()))},
                    precisions,
                    "all",
                    project)
    if len(tp_ngt) == 1:
        return
    for data_yaml, (tp, ngt) in tp_ngt.items():
        print("Results for", data_yaml, end="\n\n")
        get_aps(tp, ngt, precisions, data_yaml, project, confs)

from sys import argv
from synet.__main__ import main
def run(data_yamls : list, out_dirs : [list, None], print_jobs : bool,
        precisions : list, project : [str, None], name : None):
    """Entrypoint function.  Compute metrics of model on data_yamls.

If out_dirs is specified, it should be a list of output directories
used for validation runs on the datasets specified by data_yamls (in
the same order).

If out_dirs is not specified, then all necessary validation args
should be specified in command-line args (sys.argv).  In this case, it
will run validation of your model on each specified data yaml before
attempting to compute metrics.

If print_jobs is specified, then the commands to run the various
validation jobs are printed instead.  This is useful if you would like
to run the validation jobs in parallel.

If project is specified, this will be used as the base output
directory for plots and generated validation jobs.

name should never be specified.  validation job names are generated by
this function, so you must not try to specify your own.

precisions is a list of precision thresholds.  This is used as an
operating point which is also reported by the metrics here.  It is
either one value (used for all classes), or a list of values
correspoinding to the labels in order.

    """

    # decide output dir
    assert name is None, "--name specified by metrics.  Do not specify"
    if project is None:
        project = "metrics"
        argv.append(f"--project={project}")

    # if val was already run, just compute metrics
    if out_dirs is not None:
        assert len(out_dirs) == len(data_yamls), \
            "Please specify one output for each data yaml"
        print("Using prerun results to compute metrcis")
        return metrics(data_yamls, out_dirs, precisions, project)

    ## modify argv
    # add necessary flags
    for flag in "--save-conf", '--save-txt', '--exist-ok':
        if flag not in argv:
            argv.append(flag)
    # remove precisions flag from args
    argv.remove("--precisions")
    if print_jobs:
        argv.remove("--print-jobs")
    rm = [arg for precison in precisions for arg in argv
          if arg.isnumeric() and -.0001 <= float(arg) - precision <= .0001]
    for r in rm:
        if r in argv: argv.remove(r)
    # remove data yamls from args
    for data_yaml in data_yamls: argv.remove(data_yaml)
    # run validation
    argv.insert(1, "val")

    ## generate val jobs
    if print_jobs:
        print("Submit the following jobs:")
    out_dirs = []
    for i, data_yaml in enumerate(data_yamls):
        # specify data and out dir.
        flags = f"--data={data_yaml}", f"--name=data-split{i}"
        argv.extend(flags)
        # run/print the job
        print(" ".join(argv))
        if not print_jobs:
            print("starting job")
            main()
            # main removes job type from argv, re-add it
            argv.insert(1, "val")
        # revert argv for next job
        for flag in flags:
            argv.remove(flag)
        # keep track of output dirs in order
        out_dirs.append(join(project, f"data-split{i}"))

    ## calculate metrics
    if print_jobs:
        print("Once jobs finish, run:")

    print(" ".join([argv[0], "metrics", *data_yamls, "--out-dirs",
                    *out_dirs, "--precisions", *(str(prec)
                                                 for prec in precisions)]))
    if not print_jobs:
        print("computing metrics")
        return metrics(data_yamls, out_dirs, precisions, project)
