
"""Convenience function to load yolov5 model"""
from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression
from torch import load, no_grad, tensor
def get_yolov5_model(model_path, input_shape=(640,480), low_thld=0,
                     raw=False, **kwds):
    if model_path.endswith(".yml") or model_path.endswith(".yaml"):
        assert raw
        return Model(model_path)
    ckpt = load(model_path)['model']
    raw_model = Model(ckpt.yaml)
    raw_model.load_state_dict(ckpt.state_dict())
    if raw:
        return raw_model
    raw_model.eval()
    def model(x):
        with no_grad():
            xyxyoc = non_max_suppression(raw_model(tensor(x).unsqueeze(0).float()),
                                         conf_thres=low_thld,
                                         iou_thres=.3,
                                         multi_label=True
                                         )[0].numpy()
            return xyxyoc[:,:4], xyxyoc[:,4:].prod(1)
    return model


"""Modify Tensorflow Detect head to allow for arbitrary input shape
(need not be multiple of 32)."""
from yolov5.models.tf import TFDetect as Yolo_TFDetect
import tensorflow as tf
from tensorflow.math import ceil
class TFDetect(Yolo_TFDetect):

    # use orig __init__, but make nx, ny calculated via ceil div
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):
        super().__init__(nc, anchors, ch, imgsz, w)
        for i in range(self.nl):
            ny, nx = (ceil(self.imgsz[0] / self.stride[i]),
                      ceil(self.imgsz[1] / self.stride[i]))
            self.grid[i] = self._make_grid(nx, ny)

    # copy call method, but replace // with ceil div
    def call(self, inputs):
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = (ceil(self.imgsz[0] / self.stride[i]),
                      ceil(self.imgsz[1] / self.stride[i]))
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

            if not self.training:  # inference
                y = x[i]
                grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5
                anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3])*4
                xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]# xy
                wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]],
                                  dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]],
                                  dtype=tf.float32)
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4:5 + self.nc]),
                               y[..., 5 + self.nc:]], -1)
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

        return tf.transpose(x, [0, 2, 1, 3]) \
            if self.training else (tf.concat(z, 1),)


"""Make YOLOv5 Detect head compatible with synet tflite export"""
from .base import askeras
from yolov5.models.yolo import Detect as Yolo_PTDetect
class Detect(Yolo_PTDetect):

    def __init__(self, *args, **kwds):
        # to account for args hack.
        if len(args) == 4:
            args = args[:3]
        # construct normally
        super().__init__(*args, **kwds)
        # save args/kwargs for later construction of TF model
        self.args = args
        self.kwds = kwds

    def forward(self, x):
        if askeras.use_keras:
            return self.as_keras(x)
        return super().forward(x)

    def as_keras(self, x):
        return TFDetect(*self.args, imgsz=askeras.kwds["imgsz"],
                        w=self, **self.kwds
                        )(x)


from yolov5.val import (Path, Callbacks, create_dataloader,
                        select_device, DetectMultiBackend, check_img_size, LOGGER,
                        check_dataset, torch, np, ConfusionMatrix, coco80_to_coco91_class,
                        Profile, tqdm, non_max_suppression, scale_boxes, xywh2xyxy,
                        output_to_target, ap_per_class, pd, increment_path, os, colorstr,
                        TQDM_BAR_FORMAT, process_batch, plot_images, save_one_txt)
def val_run_tflite(
        data,
        weights=None,  # model.pt path(s)
        batch_size=None,  # batch size
        batch=None,  # batch size
        imgsz=None,  # inference size (pixels)
        img=None,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project='runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):

    if imgsz is None and img is None:
        imgsz = 640
    elif img is not None:
        imgsz = img
    if batch_size is None and batch is None:
        batch_size = 32
    elif batch is not None:
        batch_size = batch

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        half &= device.type != 'cpu' # half precision only supported on CUDA, dont remove!

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

        # SYNET MODIFICATION: check for tflite
        tflite = hasattr(model, "interpreter")

        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

        # SYNET MODIFICATION: if tflite, use that shape
        if tflite:
            sn = model.input_details[0]['shape']
            imgsz = int(max(sn[2], sn[1]))

        if not isinstance(imgsz, (list, tuple)):
            imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu' # half precision only supported on CUDA, dont remove!
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        if not isinstance(imgsz, (list, tuple)):
            model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup

        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks

        # SYNET MODIFICATION: if tflite, use rect with no padding
        if tflite:
            pad, rect = 0.0, True
            stride = np.gcd(sn[2], sn[1])

        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # SYNET MODIFICATION: if tflite, make grayscale
        if tflite:
            im = im.mean(1, keepdims=True)

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Export results as html
    header = "Class Images Labels P R mAP@.5 mAP@.5:.95"
    headers = header.split()
    data = []
    data.append(['all', seen, nt.sum(), f"{float(mp):0.3f}", f"{float(mr):0.3f}", f"{float(map50):0.3f}", f"{float(map):0.3f}"])
    for i, c in enumerate(ap_class):
        data.append([names[c], seen, nt[c], f"{float(p[i]):0.3f}", f"{float(r[i]):0.3f}", f"{float(ap50[i]):0.3f}", f"{float(ap[i]):0.3f}"])
    results_df = pd.DataFrame(data,columns=headers)
    results_html = results_df.to_html()
    text_file = open(save_dir / "results.html", "w")
    text_file.write(results_html)
    text_file.close()

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        if isinstance(imgsz, (list, tuple)):
            shape = (batch_size, 3, *imgsz)
        else:
            shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json'))  # annotations
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    map50s = np.zeros(nc) + map50
    for i, c in enumerate(ap_class):
        map50s[c] = ap50[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, map50s, t


"""Apply modifications to YOLOv5 for synet"""
from yolov5.models import yolo
from importlib import import_module
from yolov5 import val
from yolov5.models import common
from types import SimpleNamespace
import numpy
def patch_yolov5(chip=None):

    # enable the  chip if given
    if chip is not None:
        module = import_module(f"..{chip}", __name__)
        setattr(yolo, chip, module)
        yolo.Concat = module.Cat
        yolo.Detect = module.Detect = Detect

    # use modified val run function for tflites
    val.run = val_run_tflite

    # yolo uses uint8.  Change to int8
    common.np = SimpleNamespace(**vars(numpy))
    common.np.uint8 = common.np.int8
