"""Apply trained model to T1 / T2 / FLAIR NIfTI and save native-space outputs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from ailesionnet import AILesionNet, predict_mc, refine
from ailesionnet.preprocessing import load_case, restore_prediction


def main():
    parser = argparse.ArgumentParser(description='AI-LesionNet MRI inference')
    parser.add_argument('--t1', required=True)
    parser.add_argument('--t2', required=True)
    parser.add_argument('--flair', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='outputs/case')
    parser.add_argument('--shape', nargs=3, type=int, default=[32, 256, 256], metavar=('D','H','W'))
    parser.add_argument('--spacing', type=float, default=1.0)
    parser.add_argument('--no-equalize', action='store_true',
                        help='Match a checkpoint trained without histogram equalization')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--disable-refinement', action='store_true',
                        help='Ablation: threshold raw deterministic probability without uncertainty control')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.15)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-component-voxels', type=int, default=3)
    parser.add_argument('--closing-radius', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    start = time.perf_counter()
    import nibabel as nib
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = AILesionNet(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    x, _, meta = load_case(args.t1, args.t2, args.flair,
                           shape=args.shape, spacing_mm=args.spacing,
                           equalize=not args.no_equalize)
    xt = torch.from_numpy(x[None]).to(args.device)
    if args.disable_refinement:
        model.eval()
        with torch.no_grad():
            mean = torch.sigmoid(model(xt))
        variance = torch.zeros_like(mean)
    else:
        mean, variance = predict_mc(model, xt, iterations=args.iterations)
    prob = mean[0, 0].cpu().numpy()
    var = variance[0, 0].cpu().numpy()
    if args.disable_refinement:
        mask = (prob >= args.threshold).astype(np.uint8)
    else:
        mask, _, _ = refine(prob, var, alpha=args.alpha, beta=args.beta,
                            threshold=args.threshold,
                            min_component_voxels=args.min_component_voxels,
                            closing_radius=args.closing_radius)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nib.save(restore_prediction(mask, meta, order=0), str(prefix) + '_mask.nii.gz')
    nib.save(restore_prediction(prob, meta, order=1), str(prefix) + '_probability.nii.gz')
    nib.save(restore_prediction(var, meta, order=1), str(prefix) + '_uncertainty.nii.gz')
    duration = time.perf_counter() - start
    report = {'elapsed_seconds_including_preprocessing_and_mc': round(duration, 3),
              'mc_passes': 1 if args.disable_refinement else args.iterations,
              'refinement_enabled': not args.disable_refinement,
              'checkpoint_epoch': checkpoint.get('epoch'),
              'mask': str(prefix) + '_mask.nii.gz'}
    Path(str(prefix) + '_run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
