import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def describe(samples):
    v = sorted(float(x) for x in samples)
    if not v:
        raise RuntimeError('No timing samples')
    def p(q):
        pos = (len(v) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)
    return {'mean_ms': statistics.mean(v), 'median_ms': statistics.median(v), 'p05_ms': p(0.05), 'p95_ms': p(0.95), 'sd_ms': statistics.stdev(v) if len(v) > 1 else 0.0}


def synchronize(device, torch):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(fn, device, torch, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize(device, torch)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repeats):
        synchronize(device, torch)
        begin = time.perf_counter()
        fn()
        synchronize(device, torch)
        timings.append((time.perf_counter() - begin) * 1000.0)
    peak = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3) if device.type == 'cuda' else None
    return {**describe(timings), 'peak_allocated_mib': peak, 'warmup': warmup, 'repeats': repeats}, timings


def model_macs(model, x, torch):
    counts = {'conv3d': 0, 'conv_transpose3d': 0, 'linear': 0}
    handles = []
    def hook(layer, inputs, output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(layer, torch.nn.Conv3d):
            counts['conv3d'] += output.numel() * (layer.in_channels // layer.groups) * math.prod(layer.kernel_size)
        elif isinstance(layer, torch.nn.ConvTranspose3d):
            counts['conv_transpose3d'] += inputs[0].numel() * (layer.out_channels // layer.groups) * math.prod(layer.kernel_size)
        elif isinstance(layer, torch.nn.Linear):
            counts['linear'] += output.numel() * layer.in_features
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv3d, torch.nn.ConvTranspose3d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model(x)
    finally:
        for h in handles:
            h.remove()
    total = sum(counts.values())
    return counts, total, list(output.shape)


def prepare_runtime(root):
    root = Path(root).resolve()
    if not (root / 'models' / 'segresnet.py').is_file():
        raise FileNotFoundError('Corrected source root must contain models/segresnet.py')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import monai
    from models.segresnet import build_segresnet
    from prototypes.lesion_prototype import LesionPrototypeExtractor
    trainer = importlib.import_module('experiments.run_patch3_corrected_v1')
    return root, torch, monai, build_segresnet, LesionPrototypeExtractor, trainer


def load_ckpt(train_root, arm, torch):
    path = Path(train_root) / ('best_off.pt' if arm == 'A0' else 'best_on.pt')
    if not path.is_file():
        raise FileNotFoundError(str(path))
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if 'model_state' not in ck or 'proto_extractor_state' not in ck:
        raise RuntimeError('Checkpoint missing model_state or proto_extractor_state: ' + str(path))
    expected = 'off' if arm == 'A0' else 'on'
    if ck.get('arm') != expected:
        raise RuntimeError('Checkpoint arm does not match: ' + str(path))
    return ck, str(path), file_sha256(path)


def create_models(build_segresnet, LesionPrototypeExtractor, ck, device):
    net = build_segresnet(init_filters=32, in_channels=2, out_channels=1, dropout_prob=0.2, return_features=False)
    net.load_state_dict(ck['model_state'], strict=True)
    ext = LesionPrototypeExtractor(in_channels=net.get_feature_dim(), proto_dim=256)
    ext.load_state_dict(ck['proto_extractor_state'], strict=True)
    return net.to(device), ext.to(device)


def profile_static(args, runtime):
    root, torch, monai, build_net, Ext, trainer = runtime
    device = torch.device(args.device)
    outrows = []
    for arm in ('A0', 'A2'):
        ck, ckpath, cksha = load_ckpt(args.train_root, arm, torch)
        net, ext = create_models(build_net, Ext, ck, device)
        net.eval()
        net.return_features = False
        ext.eval()
        x = torch.zeros((1, 2, *args.roi_size), dtype=torch.float32, device=device)
        mac_parts, macs, outshape = model_macs(net, x, torch)
        seg_n = sum(p.numel() for p in net.parameters())
        aux_n = sum(p.numel() for p in ext.parameters())
        outrows.append({'method': arm, 'checkpoint_round': ck.get('round'), 'checkpoint_sha256': cksha,
                        'roi_size': 'x'.join(map(str, args.roi_size)), 'input_channels': 2, 'output_channels': 1,
                        'inference_params': seg_n, 'inference_params_m': seg_n / 1e6,
                        'training_auxiliary_params': aux_n, 'training_total_params': seg_n + aux_n,
                        'approx_conv_linear_gmacs': macs / 1e9,
                        'approx_conv_linear_gflops_2x': 2 * macs / 1e9,
                        'approx_conv3d_gmacs': mac_parts['conv3d'] / 1e9,
                        'approx_conv_transpose3d_gmacs': mac_parts['conv_transpose3d'] / 1e9,
                        'approx_linear_gmacs': mac_parts['linear'] / 1e9,
                        'output_shape': str(outshape)})
        del net, ext, x, ck
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    assert outrows[0]['inference_params'] == outrows[1]['inference_params'], 'Different inference parameter counts'
    assert math.isclose(outrows[0]['approx_conv_linear_gmacs'], outrows[1]['approx_conv_linear_gmacs'], abs_tol=1e-9), 'Different network ops'
    write_csv(Path(args.out) / '01_static_complexity.csv', outrows)
    print('[STATIC PASS]', json.dumps([{k: r[k] for k in ['method', 'inference_params_m', 'training_auxiliary_params', 'approx_conv_linear_gmacs']} for r in outrows], ensure_ascii=False), flush=True)


def read_val_cases(path, max_cases):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'case_id', 'split', 'dwi_path', 'adc_path', 'mask_path'}.issubset(rows[0]):
        raise ValueError('Split CSV missing required columns')
    records = []
    for row in rows:
        if row['split'].strip().lower() == 'val':
            records.append({'case_id': row['case_id'].strip(), 'dwi': row['dwi_path'].strip(),
                            'adc': row['adc_path'].strip(), 'label': row['mask_path'].strip()})
    if len(records) != 24 or len({r['case_id'] for r in records}) != 24:
        raise RuntimeError('Expected exactly 24 unique validation cases')
    records.sort(key=lambda r: r['case_id'])
    records = records[:max_cases] if max_cases > 0 else records
    for record in records:
        for key in ('dwi', 'adc', 'label'):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(f"Missing {record['case_id']} {key}: {record[key]}")
    return records


def profile_inference(args, runtime):
    root, torch, monai, build_net, Ext, trainer = runtime
    from monai.inferers import sliding_window_inference
    from datasets.transforms import get_val_transforms
    device = torch.device(args.device)
    cases = read_val_cases(args.split_csv, args.max_cases)
    tf = get_val_transforms()
    rows = []
    for arm in ('A0', 'A2'):
        ck, path, cksha = load_ckpt(args.train_root, arm, torch)
        net, ext = create_models(build_net, Ext, ck, device)
        del ext
        net.eval()
        net.return_features = False
        for index, record in enumerate(cases, 1):
            transformed = tf(record)
            x = torch.as_tensor(transformed['image']).unsqueeze(0).to(device=device, dtype=torch.float32)
            def infer():
                with torch.inference_mode():
                    logits = sliding_window_inference(x, roi_size=tuple(args.roi_size), sw_batch_size=1,
                                                     predictor=net, overlap=args.overlap)
                    return torch.sigmoid(logits) > 0.5
            timings, raw = measure(infer, device, torch, args.warmup, args.repeats)
            rows.append({'method': arm, 'case_id': record['case_id'], 'case_index': index,
                         'shape': 'x'.join(map(str, x.shape)), 'voxels': math.prod(x.shape[-3:]),
                         'roi_size': 'x'.join(map(str, args.roi_size)), 'overlap': args.overlap,
                         'checkpoint_round': ck.get('round'), 'checkpoint_sha256': cksha, **timings})
            print(f"[INFERENCE] {arm} {index}/{len(cases)} {record['case_id']} median={timings['median_ms']:.2f} ms", flush=True)
            del x, transformed
        del net, ck
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_csv(Path(args.out) / '02_full_volume_inference_cases.csv', rows)
    summaries = []
    for arm in ('A0', 'A2'):
        rr = [r for r in rows if r['method'] == arm]
        case_medians = [r['median_ms'] for r in rr]
        summaries.append({'method': arm, 'cases': len(rr), 'batch_size': 1,
                          'median_case_latency_ms': statistics.median(case_medians),
                          'mean_case_latency_ms': statistics.mean(case_medians),
                          'median_case_fps': 1000 / statistics.median(case_medians),
                          'max_peak_allocated_mib': max(r['peak_allocated_mib'] for r in rr) if device.type == 'cuda' else None,
                          'timed_scope': 'preloaded GPU input -> sliding-window segmentation mask; excludes disk and preprocessing'})
    write_csv(Path(args.out) / '03_inference_summary.csv', summaries)
    print('[INFERENCE PASS]', json.dumps(summaries, ensure_ascii=False), flush=True)


def profile_train(args, runtime):
    root, torch, monai, build_net, Ext, trainer = runtime
    from monai.losses import DiceCELoss
    from prototypes.prototype_alignment import PrototypeAlignmentLoss, ModalityConsistencyLoss
    device = torch.device(args.device)
    torch.manual_seed(17)
    x = torch.randn((args.batch_size, 2, *args.roi_size), dtype=torch.float32, device=device)
    y = torch.zeros((args.batch_size, 1, *args.roi_size), dtype=torch.float32, device=device)
    x1, x2, x3 = args.roi_size
    y[:, :, x1//3:x1//3 + max(x1//7, 1), x2//3:x2//3 + max(x2//7, 1), x3//3:x3//3 + max(x3//7, 1)] = 1
    rows = []
    for arm in ('A0', 'A2'):
        ck, ckpath, cksha = load_ckpt(args.train_root, arm, torch)
        net, ext = create_models(build_net, Ext, ck, device)
        net.return_features = True
        net.train()
        ext.train()
        optimizer = torch.optim.AdamW(list(net.parameters()) + list(ext.parameters()), lr=1e-4, weight_decay=1e-5)
        seg_fn = DiceCELoss(sigmoid=True, softmax=False, to_onehot_y=False)
        cons_fn = ModalityConsistencyLoss()
        align_fn = PrototypeAlignmentLoss()
        globals_ = {'proto_dwi': torch.zeros(256, device=device), 'proto_adc': torch.zeros(256, device=device)}
        def train_step():
            logits, features = net(x)
            seg = seg_fn(logits, y)
            proto, valid, mass = trainer.prototypes_from_batch(features, y, ext, 'soft', 'exclude', 1e-10)
            aux = seg.new_zeros(())
            if proto is not None and arm == 'A2':
                consistency = cons_fn(proto)
                alignment = align_fn(proto, globals_)
                aux = 0.005 * (0.0 * alignment + 0.05 * consistency)
            loss = seg + aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad.detach().norm() ** 2 for p in net.parameters() if p.grad is not None]
            if grads:
                grad_norm = torch.stack(grads).sum().sqrt().item()
            optimizer.step()
            return loss
        timings, raw = measure(train_step, device, torch, args.train_warmup, args.train_repeats)
        rows.append({'method': arm, 'synthetic_patch_shape': 'x'.join(map(str, args.roi_size)),
                     'batch_size': args.batch_size, 'train_stage': 'forward + loss + backward + optimizer only',
                     'optimizer': 'AdamW', 'seg_loss': 'DiceCELoss',
                     'auxiliary_loss_on': arm == 'A2', 'checkpoint_round': ck.get('round'),
                     'checkpoint_sha256': cksha, **timings})
        print(f"[TRAIN-STEP] {arm} median={timings['median_ms']:.2f} ms peak={timings['peak_allocated_mib']} MiB", flush=True)
        del optimizer, net, ext, ck
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_csv(Path(args.out) / '04_training_step.csv', rows)
    print('[TRAIN-STEP PASS]', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corrected-root', required=True)
    ap.add_argument('--train-root', required=True)
    ap.add_argument('--split-csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['static', 'inference', 'train', 'all'], default='all')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--roi-size', type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument('--overlap', type=float, default=0.5)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--train-warmup', type=int, default=3)
    ap.add_argument('--train-repeats', type=int, default=10)
    ap.add_argument('--max-cases', type=int, default=8)
    args = ap.parse_args()
    if args.device == 'cpu' and args.mode in ('all','train'):
        raise RuntimeError('Training-step GPU benchmarks require --device cuda')
    if min(args.roi_size) < 16 or args.warmup < 0 or args.repeats < 1 or args.train_repeats < 1 or args.train_warmup < 0:
        raise ValueError('Invalid input sizes or timing repetitions')
    if not 0 <= args.overlap < 1:
        raise ValueError('Invalid overlap')
    if args.mode in ('inference','all') and not args.split_csv:
        raise ValueError('--split-csv is required for full-volume inference')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = prepare_runtime(args.corrected_root)
    root, torch, monai, *_ = runtime
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in current Python environment')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    context = {'status': 'RUNNING', 'mode': args.mode, 'torch': torch.__version__, 'monai': monai.__version__,
               'cuda_available': torch.cuda.is_available(), 'device': args.device,
               'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
               'roi_size': args.roi_size, 'overlap': args.overlap, 'batch_size': args.batch_size,
               'training_folder': str(Path(args.train_root).resolve()),
               'source_model_sha256': file_sha256(root / 'models' / 'segresnet.py'),
               'timed_inference_excludes': ['disk loading', 'MONAI preprocessing', 'host-to-device copy'],
               'FLOPs_scope': '2 * MACs of Conv3d, ConvTranspose3d, Linear only; other ops excluded; approximate',
               'training_step_scope': 'fixed synthetic inputs, model forward, prototype pooling, loss, backward, AdamW; excludes dataset, augmentations, FedAvg, validation and communication'}
    try:
        if args.mode in ('static','all'):
            profile_static(args, runtime)
        if args.mode in ('inference','all'):
            profile_inference(args, runtime)
        if args.mode in ('train','all'):
            profile_train(args, runtime)
        context['status'] = 'PASS'
    except Exception as exc:
        context['status'] = 'FAIL'
        context['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with open(out / '00_environment_and_protocol.json', 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    main()
